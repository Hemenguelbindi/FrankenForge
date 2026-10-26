//! Weight delta: how far a replacing tensor is from the base tensor it replaced,
//! `||new - old||_2 / ||old||_2`. Both tensors are streamed side by side in fixed-size chunks
//! through `TensorSource::read_range`, so memory stays bounded whatever the model size.

use std::ops::Add;

use ff_core::decode::decoder;
use ff_core::{Error, Result, TensorSource};

use super::{Delta, Report};
use crate::{Payload, Plan, Sources};

/// Bytes per read from each tensor: a multiple of every element size.
const CHUNK: u64 = 4 * 1024 * 1024;

/// `Err` holds why the delta is unknown; I/O failures are the outer `Result`.
type Measured = std::result::Result<Sums, String>;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Sums {
    pub diff_sq: f64,
    pub old_sq: f64,
}

impl Add for Sums {
    type Output = Sums;
    fn add(self, o: Sums) -> Sums {
        Sums { diff_sq: self.diff_sq + o.diff_sq, old_sq: self.old_sq + o.old_sq }
    }
}

impl Sums {
    /// A zero base norm has no relative difference: Unknown, never a division by zero or 0%.
    pub fn relative(self) -> Delta {
        if self.old_sq == 0.0 {
            return Delta::Unknown { reason: "the base tensor is all zeros".into() };
        }
        Delta::Relative { value: (self.diff_sq / self.old_sq).sqrt() }
    }
}

fn unknown(reason: &str) -> Delta {
    Delta::Unknown { reason: reason.into() }
}

/// Squared sums of `new_name - old_name` and `old_name`, read chunk by chunk.
pub(crate) fn sums(
    new: &dyn TensorSource,
    new_name: &str,
    old: &dyn TensorSource,
    old_name: &str,
) -> Result<Measured> {
    let info = new.model().tensor(new_name)?;
    if !info.same_layout(old.model().tensor(old_name)?) {
        return Ok(Err("dtype or shape differs from the base tensor".into()));
    }
    let Some((size, decode)) = decoder(info.dtype) else {
        return Ok(Err(format!("{:?} values are not compared", info.dtype)));
    };
    let total = info.byte_len() as u64;
    let mut s = Sums::default();
    let mut offset = 0;
    while offset < total {
        let len = CHUNK.min(total - offset);
        let (a, b) = (new.read_range(new_name, offset, len)?, old.read_range(old_name, offset, len)?);
        if a.len() as u64 != len || b.len() as u64 != len {
            return Err(Error::Incompatible(format!("{new_name}: short read at byte {offset}")));
        }
        for (x, y) in a.chunks_exact(size).zip(b.chunks_exact(size)) {
            let (x, y) = (decode(x), decode(y));
            s.diff_sq += (x - y) * (x - y);
            s.old_sq += y * y;
        }
        offset += len;
    }
    if !(s.diff_sq.is_finite() && s.old_sq.is_finite()) {
        return Ok(Err("NaN or infinite values".into()));
    }
    Ok(Ok(s))
}

impl Report {
    /// Reads every replacing tensor and the base tensor it replaced, and fills in the deltas.
    pub fn measure(&mut self, plan: &Plan, sources: &Sources) -> Result<()> {
        let base = sources.get(&self.base)?;
        let one = |name: &str| -> Result<Measured> {
            match &plan.tensors.get(name).ok_or_else(|| Error::TensorNotFound(name.into()))?.payload {
                Payload::Source { alias, name: src } => sums(sources.get(alias)?, src, base, name),
                Payload::Generated(_) => Ok(Err("generated tensor".into())),
            }
        };
        for block in self.stacks.iter_mut().flat_map(|s| &mut s.blocks) {
            if block.replaced.is_empty() {
                continue;
            }
            let mut total: Measured = Ok(Sums::default());
            for t in &mut block.replaced {
                let m = one(&t.name)?;
                t.delta = m.as_ref().map_or_else(|r| unknown(r), |s| s.relative());
                total = match (total, m) {
                    (Ok(a), Ok(b)) => Ok(a + b),
                    (Err(e), _) => Err(e),
                    (_, Err(e)) => Err(format!("{}: {e}", t.name)),
                };
            }
            block.delta = Some(total.map_or_else(|r| unknown(&r), Sums::relative));
        }
        for o in &mut self.outside {
            if o.delta.is_some() {
                o.delta = Some(one(&o.name)?.map_or_else(|r| unknown(&r), Sums::relative));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/delta.rs"]
mod tests;
