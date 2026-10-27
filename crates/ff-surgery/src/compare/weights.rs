//! Where two models differ in their weights, block by block: `||donor - base|| / ||base||` over
//! every block both have with identical tensor names, dtypes and shapes. This reads weights,
//! streamed in bounded chunks. It says where and how far weights moved, e.g. which blocks a
//! fine-tune changed most; it says nothing about what that changes in behavior.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ff_core::{Block, Model, ModelFormat, Result, TensorSource};
use serde::Serialize;

use crate::report::delta::{Measured, Sums, combine, delta_of, sums};
use crate::report::{Delta, TensorDelta};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WeightDiff {
    pub blocks: Vec<BlockWeights>,
    /// Blocks of base that the donor lacks or has with other tensors, dtypes or shapes.
    pub not_compared: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BlockWeights {
    pub stack: String,
    pub index: usize,
    pub params: usize,
    pub delta: Delta,
    /// Per tensor, sorted by name.
    pub tensors: Vec<TensorDelta>,
}

impl BlockWeights {
    /// The tensor that moved most relative to its own size; `None` if none could be measured.
    pub fn most_changed(&self) -> Option<(&str, f64)> {
        self.tensors
            .iter()
            .filter_map(|t| match t.delta {
                Delta::Relative { value } => Some((t.name.as_str(), value)),
                _ => None,
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

/// Blocks of `base` whose tensors the donor has under the same names with the same layout, and
/// nothing else under the same prefix; plus how many base blocks did not qualify.
pub fn comparable(base: &Model, donor: &Model) -> (Vec<Block>, usize) {
    let all = base.layout().blocks;
    let total = all.len();
    let same = |b: &Block| {
        donor.select(&b.prefix).count() == b.tensors.len()
            && b.tensors
                .iter()
                .all(|n| matches!((base.tensor(n), donor.tensor(n)), (Ok(x), Ok(y)) if x.same_layout(y)))
    };
    let blocks: Vec<Block> = all.into_iter().filter(same).collect();
    let skipped = total - blocks.len();
    (blocks, skipped)
}

/// Bytes `diff` reads from both models together, for a progress bar.
pub fn bytes_to_read(base: &Model, donor: &Model) -> u64 {
    let (blocks, _) = comparable(base, donor);
    blocks.iter().flat_map(|b| &b.tensors).map(|n| base.tensors[n].byte_len() as u64 * 2).sum()
}

pub fn run(
    base: &Path,
    donor: &Path,
    format: &dyn ModelFormat,
    progress: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Option<WeightDiff>> {
    diff(&*format.open(base)?, &*format.open(donor)?, progress, stop)
}

/// `None` when `stop` was set before the end.
pub fn diff(
    base: &dyn TensorSource,
    donor: &dyn TensorSource,
    progress: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Option<WeightDiff>> {
    let (blocks, not_compared) = comparable(base.model(), donor.model());
    let mut out = Vec::new();
    for b in blocks {
        let mut total: Measured = Ok(Sums::default());
        let mut tensors = Vec::new();
        for name in &b.tensors {
            let m = sums(donor, name, base, name, progress, stop)?;
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            total = combine(total, name, &m);
            tensors.push(TensorDelta { name: name.clone(), delta: delta_of(&m) });
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let params = b.tensors.iter().map(|n| base.model().tensors[n].numel()).sum();
        out.push(BlockWeights { stack: b.stack, index: b.index, params, delta: delta_of(&total), tensors });
    }
    Ok(Some(WeightDiff { blocks: out, not_compared }))
}

impl WeightDiff {
    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self).expect("plain struct")
    }

    /// One line per block with a bar scaled to the largest block delta.
    pub fn text(&self) -> String {
        let mut s = String::from("weight difference per block, ||donor - base|| / ||base||:\n");
        let max = self.blocks.iter().filter_map(|b| relative(&b.delta)).fold(0.0, f64::max);
        let mut stack = "";
        for b in &self.blocks {
            if b.stack != stack {
                stack = &b.stack;
                let _ = writeln!(s, "{stack}");
            }
            let line = match relative(&b.delta) {
                Some(v) => {
                    let width = if max > 0.0 { (v / max * 20.0).round() as usize } else { 0 };
                    format!("{:<20}  {:>7.2}%", "#".repeat(width), v * 100.0)
                }
                None => format!("{:<20}  unknown", ""),
            };
            let most = b.most_changed().map_or_else(String::new, |(n, v)| {
                format!("   most changed: {} {:.2}%", short(n, &b.stack), v * 100.0)
            });
            let _ = writeln!(s, "  {:>3}  {line}{most}", b.index);
        }
        if self.blocks.is_empty() {
            s.push_str("no block has the same tensors, dtypes and shapes in both models\n");
        }
        if self.not_compared > 0 {
            let _ = writeln!(s, "not compared (structure differs): {} blocks", self.not_compared);
        }
        s
    }
}

/// `model.layers.15.mlp.up_proj.weight` -> `mlp.up_proj.weight`.
pub fn short<'a>(name: &'a str, stack: &str) -> &'a str {
    name.strip_prefix(stack)
        .and_then(|rest| rest.trim_start_matches('.').split_once('.'))
        .map_or(name, |(_, part)| part)
}

fn relative(d: &Delta) -> Option<f64> {
    match d {
        Delta::Relative { value } => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests/weights.rs"]
mod tests;
