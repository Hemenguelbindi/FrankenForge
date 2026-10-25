//! "What does not match": the comparison report read as a list of concrete reasons, so an
//! incompatible pair is explained instead of shown as a red cross. Reads `compare::Report` only.

use ff_surgery::compare::{Report, TokenizerMatch};
use fluent_bundle::FluentValue;

use super::analysis::tokenizer_key;
use super::schema::Side;
use crate::i18n::I18n;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mismatch {
    Tokenizer(TokenizerMatch),
    /// No repeated block structure in at least one model: the architecture is unknown here.
    NoBlocks,
    StackOnlyIn {
        stack: String,
        side: Side,
    },
    BlockCount {
        stack: String,
        base: usize,
        donor: usize,
    },
    /// Blocks at the same index whose tensors differ in shape or type; one core description shown.
    BlockShapes {
        stack: String,
        count: usize,
        index: usize,
        example: String,
    },
    /// Tensors outside the blocks (embeddings, final norm, head) with other shapes.
    OuterShapes {
        count: usize,
        example: String,
    },
    OuterMissing {
        base: usize,
        donor: usize,
    },
}

pub fn mismatches(r: &Report) -> Vec<Mismatch> {
    let mut out = Vec::new();
    if r.tokenizer.status != TokenizerMatch::Identical {
        out.push(Mismatch::Tokenizer(r.tokenizer.status));
    }
    if r.stacks.iter().all(|s| s.blocks_a == 0) || r.stacks.iter().all(|s| s.blocks_b == 0) {
        out.push(Mismatch::NoBlocks);
    }
    for s in &r.stacks {
        let stack = s.stack.clone();
        match (s.blocks_a, s.blocks_b) {
            (0, _) => out.push(Mismatch::StackOnlyIn { stack, side: Side::Donor }),
            (_, 0) => out.push(Mismatch::StackOnlyIn { stack, side: Side::Base }),
            (a, b) if a != b => out.push(Mismatch::BlockCount { stack, base: a, donor: b }),
            _ => {}
        }
        if let Some(first) = s.incompatible.first() {
            out.push(Mismatch::BlockShapes {
                stack: s.stack.clone(),
                count: s.incompatible.len(),
                index: first.index,
                example: first.differences.first().cloned().unwrap_or_default(),
            });
        }
    }
    let outer = |name: &String| !r.stacks.iter().any(|s| name.starts_with(&format!("{}.", s.stack)));
    let shapes: Vec<&String> = r.tensors.mismatched.iter().map(|m| &m.name).filter(|n| outer(n)).collect();
    if let Some(first) = shapes.first() {
        out.push(Mismatch::OuterShapes { count: shapes.len(), example: (*first).clone() });
    }
    let (base, donor) = (
        r.tensors.only_a.iter().filter(|n| outer(n)).count(),
        r.tensors.only_b.iter().filter(|n| outer(n)).count(),
    );
    if base + donor > 0 {
        out.push(Mismatch::OuterMissing { base, donor });
    }
    out
}

impl Mismatch {
    #[cfg(test)]
    pub const KEYS: [&str; 7] = [
        "forge-mismatch-no-blocks",
        "forge-mismatch-stack-only-base",
        "forge-mismatch-stack-only-donor",
        "forge-mismatch-block-count",
        "forge-mismatch-block-shapes",
        "forge-mismatch-outer-shapes",
        "forge-mismatch-outer-missing",
    ];

    /// User-facing sentence; tensor names and core descriptions are inserted as-is.
    pub fn text(&self, t: &I18n) -> String {
        let s = |v: &String| FluentValue::from(v.clone());
        match self {
            Mismatch::Tokenizer(m) => t.t(tokenizer_key(*m)),
            Mismatch::NoBlocks => t.t("forge-mismatch-no-blocks"),
            Mismatch::StackOnlyIn { stack, side } => {
                let key = match side {
                    Side::Base => "forge-mismatch-stack-only-base",
                    Side::Donor => "forge-mismatch-stack-only-donor",
                };
                t.ta(key, &[("stack", s(stack))])
            }
            Mismatch::BlockCount { stack, base, donor } => t.ta(
                "forge-mismatch-block-count",
                &[("stack", s(stack)), ("base", (*base).into()), ("donor", (*donor).into())],
            ),
            Mismatch::BlockShapes { stack, count, index, example } => t.ta(
                "forge-mismatch-block-shapes",
                &[
                    ("stack", s(stack)),
                    ("count", (*count).into()),
                    ("index", (*index).into()),
                    ("example", s(example)),
                ],
            ),
            Mismatch::OuterShapes { count, example } => {
                t.ta("forge-mismatch-outer-shapes", &[("count", (*count).into()), ("name", s(example))])
            }
            Mismatch::OuterMissing { base, donor } => {
                t.ta("forge-mismatch-outer-missing", &[("base", (*base).into()), ("donor", (*donor).into())])
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/mismatch.rs"]
mod tests;
