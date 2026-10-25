use std::collections::BTreeMap;

/// Segment names that mark a repeated stack of layers in common checkpoints
/// (`model.layers.N`, `transformer.h.N`, `blocks.N`, `encoder.layer.N`).
const STACK_KEYS: &[&str] = &["layers", "layer", "h", "blocks"];

/// One repeated unit of a stack, e.g. every tensor under `model.layers.3.`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub stack: String,
    pub index: usize,
    pub prefix: String,
    pub tensors: Vec<String>,
}

/// Blocks sorted by (stack, index) plus tensors outside any stack.
#[derive(Debug, Clone, Default)]
pub struct Layout {
    pub blocks: Vec<Block>,
    pub other: Vec<String>,
}

impl Layout {
    /// Stacks whose indices do not run 0, 1, 2, ... e.g. after removing a layer.
    pub fn gaps(&self) -> Vec<String> {
        let mut by_stack: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for b in &self.blocks {
            by_stack.entry(&b.stack).or_default().push(b.index);
        }
        by_stack
            .into_iter()
            .filter(|(_, idx)| idx.iter().enumerate().any(|(i, &n)| i != n))
            .map(|(s, idx)| format!("{s}: block indices {idx:?} are not contiguous from 0"))
            .collect()
    }
}

pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Layout {
    let mut blocks: BTreeMap<(String, usize), Block> = BTreeMap::new();
    let mut other = Vec::new();
    for name in names {
        match block_of(name) {
            Some((stack, index, prefix)) => blocks
                .entry((stack.clone(), index))
                .or_insert_with(|| Block { stack, index, prefix, tensors: Vec::new() })
                .tensors
                .push(name.to_owned()),
            None => other.push(name.to_owned()),
        }
    }
    Layout { blocks: blocks.into_values().collect(), other }
}

/// Splits `model.layers.3.mlp.up.weight` into (`model.layers`, 3, `model.layers.3.`).
pub fn block_of(name: &str) -> Option<(String, usize, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    let i = parts.windows(2).position(|w| STACK_KEYS.contains(&w[0]) && w[1].parse::<usize>().is_ok())?;
    let index = parts[i + 1].parse().ok()?;
    Some((parts[..=i].join("."), index, format!("{}.", parts[..=i + 1].join("."))))
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
