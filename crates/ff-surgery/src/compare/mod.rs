//! Metadata comparison of two models: tensors, block stacks, tokenizers.
//! Weights are not read, only headers and tokenizer.json.

pub(crate) mod tokenizer;
pub mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

use ff_core::{Block, DType, Model, ModelFormat, Result, TensorInfo};
use serde::Serialize;

pub use tokenizer::{TokenizerDiff, TokenizerMatch};

/// How many lines of each list the text report shows; the JSON has everything.
const TEXT_LIMIT: usize = 10;

#[derive(Debug, Serialize)]
pub struct Report {
    pub a: String,
    pub b: String,
    pub tensors: TensorDiff,
    pub stacks: Vec<StackDiff>,
    pub tokenizer: TokenizerDiff,
}

#[derive(Debug, Default, Serialize)]
pub struct TensorDiff {
    /// Same name, dtype and shape.
    pub same: usize,
    pub mismatched: Vec<Mismatch>,
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Spec {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct Mismatch {
    pub name: String,
    pub a: Spec,
    pub b: Spec,
}

/// A stack with the same name in A and B (or present in only one of them).
// shortcut: stacks pair by exact name (`model.layers` != `transformer.h`); add a mapping when needed.
#[derive(Debug, Serialize)]
pub struct StackDiff {
    pub stack: String,
    pub blocks_a: usize,
    pub blocks_b: usize,
    /// Indices where blocks of A and B match in composition, dtype and shape, so they can be transplanted.
    pub compatible: Vec<usize>,
    /// Blocks with the same index that do not match.
    pub incompatible: Vec<BlockDiff>,
    /// Block indices present only in A / only in B: nothing to pair them with.
    pub only_a: Vec<usize>,
    pub only_b: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct BlockDiff {
    pub index: usize,
    pub differences: Vec<String>,
}

/// Opens both models through `format` and compares them.
pub fn run(a: &Path, b: &Path, format: &dyn ModelFormat) -> Result<Report> {
    let (sa, sb) = (format.open(a)?, format.open(b)?);
    Ok(Report {
        a: a.display().to_string(),
        b: b.display().to_string(),
        tensors: tensors(sa.model(), sb.model()),
        stacks: stacks(sa.model(), sb.model()),
        tokenizer: tokenizer::compare(a, b)?,
    })
}

fn spec(t: &TensorInfo) -> Spec {
    Spec { dtype: t.dtype, shape: t.shape.clone() }
}

pub fn tensors(a: &Model, b: &Model) -> TensorDiff {
    let mut d = TensorDiff::default();
    for (name, ta) in &a.tensors {
        match b.tensors.get(name) {
            Some(tb) if ta.same_layout(tb) => d.same += 1,
            Some(tb) => d.mismatched.push(Mismatch { name: name.clone(), a: spec(ta), b: spec(tb) }),
            None => d.only_a.push(name.clone()),
        }
    }
    d.only_b = b.tensors.keys().filter(|n| !a.tensors.contains_key(*n)).cloned().collect();
    d
}

pub fn stacks(a: &Model, b: &Model) -> Vec<StackDiff> {
    let (ga, gb) = (by_stack(a), by_stack(b));
    let names: BTreeSet<&String> = ga.keys().chain(gb.keys()).collect();
    names
        .into_iter()
        .map(|stack| {
            let empty = BTreeMap::new();
            let (ba, bb) = (ga.get(stack).unwrap_or(&empty), gb.get(stack).unwrap_or(&empty));
            let mut s = StackDiff {
                stack: stack.clone(),
                blocks_a: ba.len(),
                blocks_b: bb.len(),
                compatible: Vec::new(),
                incompatible: Vec::new(),
                only_a: Vec::new(),
                only_b: bb.keys().filter(|i| !ba.contains_key(*i)).copied().collect(),
            };
            for (index, x) in ba {
                let Some(y) = bb.get(index) else {
                    s.only_a.push(*index);
                    continue;
                };
                let differences = block_diff(a, x, b, y);
                if differences.is_empty() {
                    s.compatible.push(*index);
                } else {
                    s.incompatible.push(BlockDiff { index: *index, differences });
                }
            }
            s
        })
        .collect()
}

/// stack -> index -> block.
fn by_stack(m: &Model) -> BTreeMap<String, BTreeMap<usize, Block>> {
    let mut out: BTreeMap<String, BTreeMap<usize, Block>> = BTreeMap::new();
    for b in m.layout().blocks {
        out.entry(b.stack.clone()).or_default().insert(b.index, b);
    }
    out
}

/// Block differences by relative name (`mlp.up.weight`), dtype and shape.
fn block_diff(a: &Model, x: &Block, b: &Model, y: &Block) -> Vec<String> {
    let rel = |m: &Model, blk: &Block| -> BTreeMap<String, Spec> {
        blk.tensors.iter().map(|n| (n[blk.prefix.len()..].to_owned(), spec(&m.tensors[n]))).collect()
    };
    let (ra, rb) = (rel(a, x), rel(b, y));
    let mut out = Vec::new();
    for (k, sa) in &ra {
        match rb.get(k) {
            Some(sb) if sa == sb => {}
            Some(sb) => {
                out.push(format!("{k}: {:?} {:?} vs {:?} {:?}", sa.dtype, sa.shape, sb.dtype, sb.shape))
            }
            None => out.push(format!("{k}: only in A")),
        }
    }
    out.extend(rb.keys().filter(|k| !ra.contains_key(*k)).map(|k| format!("{k}: only in B")));
    out
}

impl Report {
    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self).expect("plain structs")
    }

    pub fn text(&self) -> String {
        let mut out = String::new();
        let t = &self.tensors;
        let _ = writeln!(out, "A: {}\nB: {}", self.a, self.b);
        let _ = writeln!(
            out,
            "\ntensors: {} same, {} mismatched, {} only in A, {} only in B",
            t.same,
            t.mismatched.len(),
            t.only_a.len(),
            t.only_b.len()
        );
        let mismatched: Vec<String> = t
            .mismatched
            .iter()
            .map(|m| {
                format!("{}: A {:?} {:?}, B {:?} {:?}", m.name, m.a.dtype, m.a.shape, m.b.dtype, m.b.shape)
            })
            .collect();
        list(&mut out, "mismatched", &mismatched);
        list(&mut out, "only in A", &t.only_a);
        list(&mut out, "only in B", &t.only_b);

        for s in &self.stacks {
            let _ = writeln!(
                out,
                "\n{}: A {} blocks, B {} blocks; {} compatible, {} incompatible",
                s.stack,
                s.blocks_a,
                s.blocks_b,
                s.compatible.len(),
                s.incompatible.len()
            );
            let lines: Vec<String> = s
                .incompatible
                .iter()
                .map(|d| format!("[{}] {}", d.index, d.differences.join("; ")))
                .collect();
            list(&mut out, "incompatible", &lines);
            let indices = |v: &[usize]| v.iter().map(|i| format!("[{i}]")).collect::<Vec<_>>();
            list(&mut out, "only in A", &indices(&s.only_a));
            list(&mut out, "only in B", &indices(&s.only_b));
        }

        let tk = &self.tokenizer;
        let size = |v: Option<usize>| v.map_or("-".to_owned(), |n| n.to_string());
        let _ = writeln!(
            out,
            "\ntokenizer: {:?} (vocab A {}, B {})",
            tk.status,
            size(tk.vocab_a),
            size(tk.vocab_b)
        );
        out
    }
}

fn list(out: &mut String, title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    let _ = writeln!(out, "  {title}:");
    for item in items.iter().take(TEXT_LIMIT) {
        let _ = writeln!(out, "    {item}");
    }
    if items.len() > TEXT_LIMIT {
        let _ = writeln!(out, "    ... and {} more (see --json)", items.len() - TEXT_LIMIT);
    }
}

#[cfg(test)]
#[path = "tests/compare.rs"]
mod tests;
