//! Graph view-model of an opened checkpoint: how data flows through it. Built only from what the
//! file and config.json show (components, block stacks, tensor roles by name, shapes), so it works
//! for any architecture: whatever is not recognised is drawn as "other" with its real sizes,
//! never guessed into a known part. No egui here; `view` draws it.

pub mod view;

use serde_json::Value;

use crate::components::{Component, Kind};
use crate::inspected::Inspected;
use crate::semantics::{Group, Part, Role, classify};

#[derive(Debug, Clone, PartialEq)]
pub struct Tower {
    pub kind: Kind,
    /// Width of the stream between blocks: length of a block's norm vector, else the token
    /// embedding's last dimension.
    pub width: Option<usize>,
    /// Token and position embeddings: where data enters.
    pub inputs: Vec<Node>,
    pub stacks: Vec<StackGraph>,
    /// Norms and output heads outside the blocks: where data leaves.
    pub outputs: Vec<Node>,
    /// Projections that hand this tower's result to the language part.
    pub bridges: Vec<Node>,
    /// Tables that feed every block directly (names with `per_layer`).
    pub per_layer: Vec<Node>,
    /// Loose tensors whose purpose is not recognised.
    pub other: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub role: Role,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackGraph {
    pub name: String,
    /// Indices into `Inspected::layout.blocks`, in order.
    pub blocks: Vec<usize>,
    /// Per-block attention kind from config `layer_types`; empty unless it has one entry per block.
    pub layer_types: Vec<String>,
    /// From config `num_kv_shared_layers`: the last N blocks reuse attention memory of earlier ones.
    pub kv_shared: Option<usize>,
}

/// Language last: the other towers feed into it.
pub fn towers(inspected: &Inspected) -> Vec<Tower> {
    let mut out: Vec<Tower> = inspected.components.iter().map(|c| tower(inspected, c)).collect();
    out.sort_by_key(|t| t.kind == Kind::Language);
    out
}

fn node(inspected: &Inspected, name: &str) -> Node {
    let shape = inspected.model.tensors.get(name).map(|t| t.shape.clone()).unwrap_or_default();
    Node { name: name.to_owned(), role: classify(name).role, shape }
}

fn tower(inspected: &Inspected, c: &Component) -> Tower {
    let mut t = Tower {
        kind: c.kind,
        width: None,
        inputs: Vec::new(),
        stacks: Vec::new(),
        outputs: Vec::new(),
        bridges: Vec::new(),
        per_layer: Vec::new(),
        other: 0,
    };
    for name in &c.loose {
        let n = node(inspected, name);
        match n.role {
            _ if name.contains("per_layer") => t.per_layer.push(n),
            Role::TokenEmbedding | Role::PositionEmbedding => t.inputs.push(n),
            Role::Projector => t.bridges.push(n),
            Role::OutputHead | Role::FinalNorm | Role::Norm => t.outputs.push(n),
            _ => t.other += 1,
        }
    }
    let config = if c.kind == Kind::Language { text_config(inspected.config_value.as_ref()) } else { None };
    for s in &c.stacks {
        let n = s.blocks.len();
        let layer_types: Vec<String> = config
            .and_then(|v| v.get("layer_types")?.as_array().cloned())
            .filter(|a| a.len() == n)
            .map(|a| a.iter().map(|x| x.as_str().unwrap_or("?").to_owned()).collect())
            .unwrap_or_default();
        let kv_shared = config
            .and_then(|v| v.get("num_kv_shared_layers")?.as_u64())
            .map(|k| k as usize)
            .filter(|&k| k > 0 && k < n);
        t.stacks.push(StackGraph { name: s.name.clone(), blocks: s.blocks.clone(), layer_types, kv_shared });
    }
    t.width = c
        .stacks
        .first()
        .and_then(|s| s.blocks.first())
        .and_then(|&i| block_width(inspected, i))
        .or_else(|| t.inputs.iter().find(|n| n.role == Role::TokenEmbedding)?.shape.get(1).copied());
    t
}

/// The text part's settings: the root of a text model, `text_config` of a multimodal one.
fn text_config(config: Option<&Value>) -> Option<&Value> {
    let v = config?;
    v.get("text_config").or(Some(v))
}

/// The most common length of the block's 1-D norm vectors (q/k norms have the head width).
fn block_width(inspected: &Inspected, block: usize) -> Option<usize> {
    let mut lengths: Vec<usize> = inspected.layout.blocks[block]
        .tensors
        .iter()
        .filter(|n| matches!(classify(n).role, Role::Norm))
        .filter_map(|n| match inspected.model.tensors[n].shape[..] {
            [len] => Some(len),
            _ => None,
        })
        .collect();
    lengths.sort_unstable();
    let mut best: Option<(usize, usize)> = None;
    for chunk in lengths.chunk_by(|a, b| a == b) {
        if best.is_none_or(|(_, n)| chunk.len() > n) {
            best = Some((chunk[0], chunk.len()));
        }
    }
    best.map(|(len, _)| len)
}

/// One block drawn as branches off the stream: each reads it and adds its result back.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockGraph {
    pub width: Option<usize>,
    pub branches: Vec<Branch>,
    /// Norm vectors in the block; they rescale numbers between steps.
    pub norms: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub group: Group,
    pub matrices: Vec<Matrix>,
    /// Scalars, biases and other non-matrix parts of the branch.
    pub small: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Full name of the first tensor with this role, so a click can open it.
    pub name: String,
    pub role: Role,
    /// Tensors with the same role (experts); 1 otherwise.
    pub count: usize,
    pub flow: Flow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flow {
    /// Reads the stream (`from` numbers) and produces `to`.
    Reads { from: usize, to: usize },
    /// Produces the stream's `to` numbers from `from`.
    Writes { from: usize, to: usize },
    /// No dimension equals the stream width: shown as stored.
    Shape(Vec<usize>),
}

/// Matrices that write back into the stream; every other recognised matrix reads from it.
fn writes(role: Role) -> bool {
    matches!(
        role,
        Role::AttnO | Role::Mlp(Part::Down) | Role::Expert(Part::Down) | Role::SharedExpert(Part::Down)
    )
}

/// The direction comes from the role, not the storage order, so transposed layouts
/// (GPT-2's `[in, out]`) read the same as `[out, in]`.
pub fn flow(role: Role, shape: &[usize], width: Option<usize>) -> Flow {
    let other = |w: usize| match shape {
        [a, b] if *a == w => Some(*b),
        [a, b] if *b == w => Some(*a),
        _ => None,
    };
    match width.and_then(|w| Some((w, other(w)?))) {
        Some((w, o)) if writes(role) => Flow::Writes { from: o, to: w },
        // An unrecognised `[w, x]` most likely writes `x -> w` (`[out, in]` storage).
        Some((w, o)) if role == Role::Unknown && shape[0] == w && shape[1] != w => {
            Flow::Writes { from: o, to: w }
        }
        Some((w, o)) => Flow::Reads { from: w, to: o },
        None => Flow::Shape(shape.to_vec()),
    }
}

/// Branch order: attention, then feed-forward, experts, the rest.
const ORDER: [Group; 4] = [Group::Attention, Group::Mlp, Group::Experts, Group::Other];

pub fn block(inspected: &Inspected, index: usize, width: Option<usize>) -> BlockGraph {
    let block = &inspected.layout.blocks[index];
    let mut branches: Vec<Branch> =
        ORDER.iter().map(|&group| Branch { group, matrices: Vec::new(), small: 0 }).collect();
    let mut norms = 0;
    for name in &block.tensors {
        let role = classify(name).role;
        let shape = &inspected.model.tensors[name].shape;
        if role.group() == Group::Normalization {
            norms += 1;
            continue;
        }
        let b =
            branches.iter_mut().find(|b| b.group == role.group()).expect("ORDER covers every other group");
        if shape.len() != 2 {
            b.small += 1;
            continue;
        }
        // Experts repeat one role many times; unknown parts are each their own row.
        match b.matrices.iter_mut().find(|m| m.role == role && role != Role::Unknown) {
            Some(m) => m.count += 1,
            None => {
                b.matrices.push(Matrix { name: name.clone(), role, count: 1, flow: flow(role, shape, width) })
            }
        }
    }
    // Within a branch, the order data takes: what reads the stream first, what writes back last.
    for b in &mut branches {
        b.matrices.sort_by_key(|m| matches!(m.flow, Flow::Writes { .. }));
    }
    branches.retain(|b| !b.matrices.is_empty() || b.small > 0);
    BlockGraph { width, branches, norms }
}

#[cfg(test)]
#[path = "tests/graph.rs"]
mod tests;
