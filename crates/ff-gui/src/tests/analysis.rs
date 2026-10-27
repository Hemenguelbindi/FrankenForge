//! Tests for module `analysis`: synthetic tensors with precomputed statistics, every supported
//! dtype, NaN/Inf, empty and damaged data, bounded streaming, and no reads before analysis.

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use super::*;
use ff_core::{Model, ModelFormat, Result as CoreResult};
use ff_io::Safetensors;

use crate::inspected::Inspected;
use crate::semantics::describe;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 keeps the top 16 bits of an f32; the test values are exactly representable.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run(dtype: DType, bytes: &[u8]) -> Result<Analysis, Failure> {
    let mut read = |s: u64, l: u64| Ok(bytes[s as usize..(s + l) as usize].to_vec());
    profile(dtype, bytes.len() as u64, &mut read, &AtomicU64::new(0), &AtomicBool::new(false))
}

fn finite(a: Result<Analysis, Failure>) -> (Profile, Finite) {
    let Ok(Analysis::Profile(p)) = a else { panic!("{a:?}") };
    let f = p.finite.clone().expect("finite values");
    (p, f)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn f32_profile_matches_precomputed_statistics() {
    let (p, f) = finite(run(DType::F32, &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, -1.0, 5.0])));
    assert_eq!((p.count, p.zeros, p.nan, p.pos_inf, p.neg_inf), (8, 2, 0, 0, 0));
    assert_eq!((f.min, f.max, f.count), (-1.0, 5.0, 8));
    assert!(close(f.mean, 1.75), "{}", f.mean);
    // Population variance: sum of squared deviations 31.5 over 8 values.
    assert!(close(f.std, (31.5f64 / 8.0).sqrt()), "{}", f.std);
    assert_eq!(f.histogram.iter().sum::<u64>(), 8);
    assert_eq!((f.histogram[0], f.histogram[BINS - 1]), (1, 1), "min and max land in the end bins");
}

#[test]
fn bf16_and_f16_decode_exactly() {
    let (_, f) = finite(run(DType::BF16, &bf16_bytes(&[1.0, -2.0, 0.5])));
    assert_eq!((f.min, f.max), (-2.0, 1.0));
    assert!(close(f.mean, -0.5 / 3.0));

    let (p, f) = finite(run(DType::F16, &u16_bytes(&[0x3C00, 0xC000, 0x0000, 0x7C00, 0x7E00])));
    assert_eq!((p.count, p.zeros, p.nan, p.pos_inf), (5, 1, 1, 1));
    assert_eq!((f.count, f.min, f.max), (3, -2.0, 1.0));
}

#[test]
fn nan_and_inf_are_counted_and_kept_out_of_the_moments() {
    let (p, f) = finite(run(DType::F32, &f32_bytes(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 3.0])));
    assert_eq!((p.nan, p.pos_inf, p.neg_inf), (1, 1, 1));
    assert_eq!((f.count, f.min, f.max), (2, 1.0, 3.0));
    assert!(close(f.mean, 2.0) && close(f.std, 1.0));
    // Nothing finite at all: no moments and no histogram.
    let Ok(Analysis::Profile(p)) = run(DType::F32, &f32_bytes(&[f32::NAN, f32::INFINITY])) else { panic!() };
    assert_eq!((p.count, p.finite), (2, None));
}

#[test]
fn empty_unsupported_and_damaged_data() {
    assert_eq!(run(DType::F32, &[]), Ok(Analysis::Empty));
    // An integer tensor is not read at all.
    let reads = Cell::new(0);
    let mut read = |_: u64, _: u64| {
        reads.set(reads.get() + 1);
        Ok(vec![])
    };
    let r = profile(DType::U8, 16, &mut read, &AtomicU64::new(0), &AtomicBool::new(false));
    assert_eq!((r, reads.get()), (Ok(Analysis::Unsupported(DType::U8)), 0));
    // Not a whole number of elements.
    assert_eq!(run(DType::F32, &[0; 6]).map_err(|f| f.kind), Err(FailureKind::Format));
    // The source returns fewer bytes than asked for.
    let mut short = |_: u64, l: u64| Ok(vec![0; l as usize - 1]);
    let r = profile(DType::F32, 8, &mut short, &AtomicU64::new(0), &AtomicBool::new(false));
    assert_eq!(r.map_err(|f| f.kind), Err(FailureKind::Format));
    // The source fails: the core's error is kept as the technical detail.
    let mut broken = |_: u64, _: u64| -> CoreResult<Vec<u8>> { Err(ff_core::Error::Verify("disk".into())) };
    let r = profile(DType::F32, 8, &mut broken, &AtomicU64::new(0), &AtomicBool::new(false));
    assert!(r.is_err_and(|f| f.detail.contains("disk")));
}

#[test]
fn large_tensors_are_streamed_in_bounded_chunks() {
    let total = 3 * CHUNK + 8;
    let (largest, calls) = (Cell::new(0u64), Cell::new(0u32));
    let mut read = |_: u64, l: u64| {
        largest.set(largest.get().max(l));
        calls.set(calls.get() + 1);
        Ok(vec![0u8; l as usize])
    };
    let progress = AtomicU64::new(0);
    let r = profile(DType::F32, total, &mut read, &progress, &AtomicBool::new(false));
    let Ok(Analysis::Profile(p)) = r else { panic!("{r:?}") };
    assert_eq!((p.count, p.zeros), (total / 4, total / 4));
    assert_eq!(largest.get(), CHUNK, "never more than one chunk in memory");
    assert_eq!(calls.get(), 8, "two passes of four chunks");
    assert_eq!(progress.load(Ordering::Relaxed), 2 * total);
}

#[test]
fn stop_flag_cancels_between_chunks() {
    let stop = AtomicBool::new(true);
    let mut read = |_: u64, l: u64| Ok(vec![0u8; l as usize]);
    let r = profile(DType::F32, 8, &mut read, &AtomicU64::new(0), &stop);
    assert_eq!(r.map_err(|f| f.kind), Err(FailureKind::Cancelled));
}

#[test]
fn extreme_values_do_not_overflow() {
    let (_, f) = finite(run(DType::F32, &f32_bytes(&[f32::MIN, 0.0, f32::MAX])));
    assert_eq!((f.histogram[0], f.histogram[BINS / 2], f.histogram[BINS - 1]), (1, 1, 1));
    assert!(f.mean.is_finite() && f.std.is_finite(), "{f:?}");
    // F64 extremes would overflow the accumulators, so F64 is reported as unsupported.
    assert_eq!(run(DType::F64, &[0; 8]), Ok(Analysis::Unsupported(DType::F64)));
}

#[test]
fn constant_tensor_lands_in_the_first_bin() {
    let (p, f) = finite(run(DType::F32, &f32_bytes(&[2.0; 4])));
    assert_eq!((f.min, f.max, f.mean, f.std), (2.0, 2.0, 2.0, 0.0));
    assert_eq!(f.histogram[0], 4, "a zero-width range puts every value in bin 0");
    assert_eq!(f.histogram[1..].iter().sum::<u64>(), 0);
    assert_eq!(p.zeros, 0);
}

#[test]
fn an_overflowing_shape_has_no_byte_length() {
    let info = TensorInfo { name: "x".into(), dtype: DType::F32, shape: vec![usize::MAX, 4] };
    assert_eq!(byte_len(&info), None, "checked multiplication must not wrap");
}

/// Counts every weight read that goes through the port.
struct Counting(Arc<AtomicUsize>);
struct CountingSource(Box<dyn TensorSource>, Arc<AtomicUsize>);

impl ModelFormat for Counting {
    fn open(&self, path: &Path) -> CoreResult<Box<dyn TensorSource>> {
        Ok(Box::new(CountingSource(Safetensors.open(path)?, self.0.clone())))
    }
    fn file_name(&self) -> &'static str {
        Safetensors.file_name()
    }
    fn write(
        &self,
        path: &Path,
        model: &Model,
        fetch: &mut dyn FnMut(&TensorInfo) -> CoreResult<Vec<u8>>,
    ) -> CoreResult<()> {
        Safetensors.write(path, model, fetch)
    }
}

impl TensorSource for CountingSource {
    fn model(&self) -> &Model {
        self.0.model()
    }
    fn read(&self, name: &str) -> CoreResult<Vec<u8>> {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.read(name)
    }
    fn read_range(&self, name: &str, start: u64, len: u64) -> CoreResult<Vec<u8>> {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.read_range(name, start, len)
    }
}

#[test]
fn weights_are_read_only_when_analysis_is_requested() {
    let dir: PathBuf = std::env::temp_dir().join(format!("ff-gui-analysis-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut m = Model::default();
    m.insert(TensorInfo {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        dtype: DType::F32,
        shape: vec![2, 2],
    });
    m.insert(TensorInfo { name: "model.embed_tokens.weight".into(), dtype: DType::BF16, shape: vec![4, 2] });
    Safetensors
        .write(&dir.join("model.safetensors"), &m, &mut |t| match t.dtype {
            DType::F32 => Ok(f32_bytes(&[1.0, -1.0, 0.0, 2.0])),
            _ => Ok(bf16_bytes(&[0.5; 8])),
        })
        .unwrap();

    let reads = Arc::new(AtomicUsize::new(0));
    let format = Counting(reads.clone());
    // Opening, grouping and explaining every tensor touch metadata only.
    let inspected = Inspected::open(&dir, &format).unwrap();
    for info in inspected.model.tensors.values() {
        let _ = describe(info, inspected.config_ref());
    }
    assert_eq!(reads.load(Ordering::Relaxed), 0, "no weight read before Analyze");

    let info = &inspected.model.tensors["model.layers.0.self_attn.q_proj.weight"];
    let (progress, stop) = (AtomicU64::new(0), AtomicBool::new(false));
    let (p, f) = finite(analyze(&*inspected.source, info, &progress, &stop));
    assert_eq!((p.count, p.zeros, f.min, f.max), (4, 1, -1.0, 2.0));
    assert!(reads.load(Ordering::Relaxed) > 0);

    // Bytes are never decoded with a dtype or shape the source does not report.
    let stale = TensorInfo { dtype: DType::F16, shape: vec![2, 4], ..info.clone() };
    let before = reads.load(Ordering::Relaxed);
    let r = analyze(&*inspected.source, &stale, &progress, &stop);
    assert_eq!(r.map_err(|f| f.kind), Err(FailureKind::Format));
    assert_eq!(reads.load(Ordering::Relaxed), before, "refused before reading");
    fs::remove_dir_all(&dir).unwrap();
}
