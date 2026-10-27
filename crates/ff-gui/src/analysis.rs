//! Numeric profile of one tensor, computed only when the user asks. The tensor is streamed in
//! fixed-size chunks over two passes (moments, then a histogram over the observed range), so
//! memory stays bounded however large the tensor is. Nothing here says what the weights mean.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ff_core::decode::{Decode, decoder};
use ff_core::{DType, TensorInfo, TensorSource};

use crate::failure::{Failure, FailureKind};

pub const BINS: usize = 40;
/// Bytes per read: a multiple of every supported element size; large enough that reopening the
/// file per read stays cheap, small enough to keep memory bounded.
const CHUNK: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum Analysis {
    Profile(Profile),
    Empty,
    Unsupported(DType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub count: u64,
    pub zeros: u64,
    pub nan: u64,
    pub pos_inf: u64,
    pub neg_inf: u64,
    /// Statistics over finite values; `None` when every value is NaN or infinite.
    pub finite: Option<Finite>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Finite {
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
    /// `BINS` equal-width bins over `[min, max]`.
    pub histogram: Vec<u64>,
}

/// Byte length from the shape with overflow checks, independent of the header's offsets.
pub fn byte_len(info: &TensorInfo) -> Option<u64> {
    info.shape.iter().try_fold(info.dtype.size_bytes() as u64, |acc, &d| acc.checked_mul(d as u64))
}

/// Streams one tensor from the source its metadata came from. `progress` counts processed
/// bytes over both passes (`2 * byte_len`); setting `stop` ends the work between chunks.
pub fn analyze(
    source: &dyn TensorSource,
    info: &TensorInfo,
    progress: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Analysis, Failure> {
    // The bytes are decoded with `info`; refuse if the source no longer describes them that way.
    let current = source.model().tensor(&info.name).map_err(|e| Failure::from_error(&e))?;
    if !current.same_layout(info) {
        return Err(corrupt(format!("{}: dtype or shape differs from what is shown", info.name)));
    }
    // shortcut: the two passes assume the file does not change between them; add a size/mtime
    // stamp if checkpoints are ever analyzed while another process writes them.
    let total = byte_len(info).ok_or_else(|| corrupt(format!("{}: size overflows", info.name)))?;
    profile(info.dtype, total, &mut |start, len| source.read_range(&info.name, start, len), progress, stop)
}

/// The profile of `total` bytes of `dtype` delivered by `read(start, len)`.
pub fn profile(
    dtype: DType,
    total: u64,
    read: &mut dyn FnMut(u64, u64) -> ff_core::Result<Vec<u8>>,
    progress: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Analysis, Failure> {
    let Some((size, decode)) = decoder(dtype) else { return Ok(Analysis::Unsupported(dtype)) };
    if total == 0 {
        return Ok(Analysis::Empty);
    }
    if !total.is_multiple_of(size as u64) {
        return Err(corrupt(format!("{total} bytes is not a whole number of {size}-byte elements")));
    }
    let mut moments = Moments::default();
    stream(total, size, decode, read, progress, stop, |v| moments.push(v))?;
    let mut profile = moments.finish();
    if let Some(finite) = &mut profile.finite {
        let (lo, hi) = (finite.min, finite.max);
        let bins = &mut finite.histogram;
        stream(total, size, decode, read, progress, stop, |v| {
            if v.is_finite() {
                bins[bin(v, lo, hi)] += 1;
            }
        })?;
    }
    Ok(Analysis::Profile(profile))
}

fn stream(
    total: u64,
    size: usize,
    decode: Decode,
    read: &mut dyn FnMut(u64, u64) -> ff_core::Result<Vec<u8>>,
    progress: &AtomicU64,
    stop: &AtomicBool,
    mut each: impl FnMut(f64),
) -> Result<(), Failure> {
    let mut offset = 0u64;
    while offset < total {
        if stop.load(Ordering::Relaxed) {
            return Err(Failure {
                kind: FailureKind::Cancelled,
                detail: "superseded by another request".into(),
            });
        }
        // `offset < total`, so neither the subtraction nor the addition below can overflow.
        let len = CHUNK.min(total - offset);
        let bytes = read(offset, len).map_err(|e| Failure::from_error(&e))?;
        if bytes.len() as u64 != len {
            return Err(corrupt(format!("asked for {len} bytes at {offset}, got {}", bytes.len())));
        }
        for element in bytes.chunks_exact(size) {
            each(decode(element));
        }
        offset += len;
        progress.fetch_add(len, Ordering::Relaxed);
    }
    Ok(())
}

fn corrupt(detail: String) -> Failure {
    Failure { kind: FailureKind::Format, detail }
}

/// Bin of a finite `v` in `[lo, hi]`. Halving first keeps `hi - lo` finite even for f64 extremes.
fn bin(v: f64, lo: f64, hi: f64) -> usize {
    let width = hi / 2.0 - lo / 2.0;
    if width <= 0.0 {
        return 0;
    }
    let i = ((v / 2.0 - lo / 2.0) / width * BINS as f64) as usize;
    i.min(BINS - 1)
}

/// Welford's running mean and variance in f64: stable over billions of values, and for the
/// 16/32-bit inputs `decoder` accepts no intermediate can overflow (|x| < 3.5e38, so squares stay
/// far below f64::MAX).
#[derive(Default)]
struct Moments {
    count: u64,
    zeros: u64,
    nan: u64,
    pos_inf: u64,
    neg_inf: u64,
    finite: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Moments {
    fn push(&mut self, v: f64) {
        // Element counts come from a validated header, far below u64::MAX.
        self.count += 1;
        if v.is_nan() {
            self.nan += 1;
            return;
        }
        if v.is_infinite() {
            if v > 0.0 {
                self.pos_inf += 1
            } else {
                self.neg_inf += 1
            }
            return;
        }
        if v == 0.0 {
            self.zeros += 1;
        }
        self.finite += 1;
        if self.finite == 1 {
            (self.min, self.max) = (v, v);
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        let delta = v - self.mean;
        self.mean += delta / self.finite as f64;
        self.m2 += delta * (v - self.mean);
    }

    fn finish(self) -> Profile {
        let finite = (self.finite > 0).then(|| Finite {
            count: self.finite,
            min: self.min,
            max: self.max,
            mean: self.mean,
            std: (self.m2 / self.finite as f64).max(0.0).sqrt(),
            histogram: vec![0; BINS],
        });
        Profile {
            count: self.count,
            zeros: self.zeros,
            nan: self.nan,
            pos_inf: self.pos_inf,
            neg_inf: self.neg_inf,
            finite,
        }
    }
}

#[cfg(test)]
#[path = "tests/analysis.rs"]
mod tests;
