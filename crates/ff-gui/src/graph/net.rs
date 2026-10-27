//! The model as nodes and links, for the free-form graph: one node per table, block, bridge and
//! output, sized by its parameters, linked along the path data takes. Which node is the largest,
//! where data enters and leaves, and which node has the most links are found here, not by eye.

use crate::components::Kind;
use crate::inspected::Inspected;
use crate::semantics::Role;

use super::{Node, Tower};

#[derive(Debug, Clone, PartialEq)]
pub enum What {
    /// A table with one row per token: input words or the output head.
    Words(Node),
    /// A table that hands every block its own input per token (`per_layer`).
    PerLayer {
        node: Node,
        blocks: usize,
    },
    Block {
        layout: usize,
        stack: String,
        index: usize,
        shared_memory: bool,
        kind: Option<String>,
    },
    /// Hands a tower's result to the language part.
    Bridge(Node),
    /// Final norm or another part after the blocks.
    Output(Node),
    /// Position table or another input that is not a word table.
    Input(Node),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetNode {
    pub what: What,
    pub tower: Kind,
    pub params: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// Data passes from one part to the next.
    Flow,
    /// A per-layer table feeding a block directly.
    Feed,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Net {
    pub nodes: Vec<NetNode>,
    pub links: Vec<(usize, usize, Link)>,
}

/// Nodes the summary names; `None` when the model has no such part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landmarks {
    pub largest: Option<usize>,
    pub entry: Option<usize>,
    pub exit: Option<usize>,
    pub most_linked: Option<usize>,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Net {
    fn add(&mut self, what: What, tower: Kind, params: usize) -> usize {
        self.nodes.push(NetNode { what, tower, params });
        self.nodes.len() - 1
    }

    fn link(&mut self, from: usize, to: usize, link: Link) {
        self.links.push((from, to, link));
    }
}

pub fn build(inspected: &Inspected, towers: &[Tower]) -> Net {
    let mut net = Net::default();
    // Language comes last in `towers`, so bridges are linked to its first block at the end.
    let mut bridges = Vec::new();
    let mut language_first = None;
    for t in towers {
        let inputs: Vec<usize> = t
            .inputs
            .iter()
            .map(|n| {
                let what = if n.role == Role::TokenEmbedding {
                    What::Words(n.clone())
                } else {
                    What::Input(n.clone())
                };
                net.add(what, t.kind, numel(&n.shape))
            })
            .collect();
        let mut blocks = Vec::new();
        for s in &t.stacks {
            let shared_from = s.kv_shared.map(|k| s.blocks.len() - k);
            for (pos, &layout) in s.blocks.iter().enumerate() {
                let b = &inspected.layout.blocks[layout];
                let params = b.tensors.iter().map(|n| inspected.model.tensors[n].numel()).sum();
                let what = What::Block {
                    layout,
                    stack: s.name.clone(),
                    index: b.index,
                    shared_memory: shared_from.is_some_and(|f| pos >= f),
                    kind: s.layer_types.get(pos).cloned(),
                };
                blocks.push(net.add(what, t.kind, params));
            }
        }
        for pair in blocks.windows(2) {
            net.link(pair[0], pair[1], Link::Flow);
        }
        if let Some(&first) = blocks.first() {
            inputs.iter().for_each(|&i| net.link(i, first, Link::Flow));
            if t.kind == Kind::Language {
                language_first = Some(first);
            }
        }
        for n in &t.per_layer {
            let i =
                net.add(What::PerLayer { node: n.clone(), blocks: blocks.len() }, t.kind, numel(&n.shape));
            if n.shape.len() == 2 {
                blocks.iter().for_each(|&b| net.link(i, b, Link::Feed));
            }
        }
        let mut last = blocks.last().or(inputs.last()).copied();
        let after = t.outputs.iter().map(|n| {
            let what =
                if n.role == Role::OutputHead { What::Words(n.clone()) } else { What::Output(n.clone()) };
            (what, n)
        });
        let after: Vec<(What, &Node)> =
            after.chain(t.bridges.iter().map(|n| (What::Bridge(n.clone()), n))).collect();
        for (what, n) in after {
            let bridge = matches!(what, What::Bridge(_));
            let i = net.add(what, t.kind, numel(&n.shape));
            if let Some(prev) = last {
                net.link(prev, i, Link::Flow);
            }
            last = Some(i);
            if bridge {
                bridges.push(i);
            }
        }
    }
    if let Some(first) = language_first {
        bridges.into_iter().for_each(|b| net.link(b, first, Link::Flow));
    }
    net
}

pub fn landmarks(net: &Net) -> Landmarks {
    let mut degree = vec![0usize; net.nodes.len()];
    for &(a, b, _) in &net.links {
        degree[a] += 1;
        degree[b] += 1;
    }
    let words = |n: &NetNode, role: Role| matches!(&n.what, What::Words(w) if w.role == role);
    let language = |n: &NetNode| n.tower == Kind::Language;
    let entry = net.nodes.iter().position(|n| language(n) && words(n, Role::TokenEmbedding));
    let exit =
        net.nodes.iter().rposition(|n| language(n) && matches!(n.what, What::Words(_) | What::Output(_)));
    Landmarks {
        largest: (0..net.nodes.len()).max_by_key(|&i| net.nodes[i].params),
        entry: entry
            .or_else(|| net.nodes.iter().position(|n| matches!(n.what, What::Words(_) | What::Input(_)))),
        exit: exit.or_else(|| net.nodes.iter().rposition(|n| matches!(n.what, What::Output(_)))),
        // A chain gives every block two links; only a node with more is a hub worth naming.
        most_linked: (0..net.nodes.len()).filter(|&i| degree[i] > 2).max_by_key(|&i| degree[i]),
    }
}

#[cfg(test)]
#[path = "tests/net.rs"]
mod tests;
