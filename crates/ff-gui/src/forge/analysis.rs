//! The comparison report as Forge shows it: one row per block index found in either model.
//! Which blocks fit is decided by `ff_surgery::compare`; this module only lays its lists out.

use ff_surgery::compare::{Report, TokenizerMatch};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId {
    pub stack: String,
    pub index: usize,
}

impl BlockId {
    /// Block prefix selector (ends in `.`, so `layers.1.` never matches `layers.10.`).
    pub fn prefix(&self) -> String {
        format!("{}.{}.", self.stack, self.index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockStatus {
    /// Same tensor names, types and shapes in both models: structurally compatible.
    Compatible,
    /// Structure differs; the core's own descriptions of each difference.
    Different(Vec<String>),
    /// Nothing at this index in the Donor, so there is nothing to take.
    OnlyInBase,
    /// The Base has no block at this index to replace.
    OnlyInDonor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRow {
    pub id: BlockId,
    pub status: BlockStatus,
}

pub struct Analysis {
    pub report: Report,
    pub rows: Vec<BlockRow>,
}

impl Analysis {
    /// Report A is the Base, B the Donor.
    pub fn new(report: Report) -> Self {
        let mut rows = Vec::new();
        for s in &report.stacks {
            let row =
                |index: usize, status| BlockRow { id: BlockId { stack: s.stack.clone(), index }, status };
            let start = rows.len();
            rows.extend(s.compatible.iter().map(|&i| row(i, BlockStatus::Compatible)));
            rows.extend(
                s.incompatible.iter().map(|d| row(d.index, BlockStatus::Different(d.differences.clone()))),
            );
            rows.extend(s.only_a.iter().map(|&i| row(i, BlockStatus::OnlyInBase)));
            rows.extend(s.only_b.iter().map(|&i| row(i, BlockStatus::OnlyInDonor)));
            rows[start..].sort_by_key(|r| r.id.index);
        }
        Self { report, rows }
    }

    pub fn compatible(&self, id: &BlockId) -> bool {
        self.rows.iter().any(|r| &r.id == id && r.status == BlockStatus::Compatible)
    }
}

pub fn tokenizer_key(m: TokenizerMatch) -> &'static str {
    match m {
        TokenizerMatch::Identical => "forge-tokenizer-identical",
        TokenizerMatch::SameVocab => "forge-tokenizer-same-vocab",
        TokenizerMatch::DifferentVocab => "forge-tokenizer-different-vocab",
        TokenizerMatch::Missing => "forge-tokenizer-missing",
    }
}

#[cfg(test)]
#[path = "tests/analysis.rs"]
mod tests;
