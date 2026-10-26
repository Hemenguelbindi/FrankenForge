//! Human-readable account of what a plan does to the base model, block by block: where each
//! output block comes from, what was removed, what changed in config.json, and how far
//! replaced weights moved. Built from metadata; only `measure` reads weights. The report never
//! claims anything about model behavior: nothing here runs the model.

pub(crate) mod delta;
mod text;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, Model, Result, layout};
use serde::{Deserialize, Serialize};

use crate::companion::CONFIG;
use crate::plan::Experimental;
use crate::{Payload, Plan};

pub use text::Row;

/// Bumped when a field changes meaning or disappears; new optional fields keep the version.
pub const SCHEMA_VERSION: u32 = 1;
/// The only value `model_behavior` ever has.
pub const NOT_EVALUATED: &str = "not_evaluated";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    /// Alias of the base source every status is relative to.
    pub base: String,
    pub stacks: Vec<Stack>,
    /// Base blocks none of whose tensors reach the output.
    pub removed_blocks: Vec<BlockRef>,
    /// Base tensors that do not reach the output, outside the removed blocks.
    pub removed_tensors: Vec<String>,
    /// Output tensors outside any stack.
    pub outside: Vec<Outside>,
    /// Whether the base has a config.json at all; `config_changes` is empty when it is copied as is.
    pub config_present: bool,
    pub config_changes: Vec<String>,
    pub warnings: Vec<String>,
    pub experimental: Option<Experimental>,
    pub file: FileCheck,
    /// Always [`NOT_EVALUATED`]: no inference runs, so nothing is known about behavior.
    pub model_behavior: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileCheck {
    /// `plan`: nothing was written.
    NotWritten,
    /// `build`: every tensor was re-read from the output and its hash matched.
    Verified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stack {
    pub name: String,
    pub blocks_before: usize,
    pub blocks_after: usize,
    pub blocks: Vec<BlockReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockReport {
    pub index: usize,
    pub tensors: usize,
    /// Distinct (alias, source block) pairs the block's tensors come from.
    pub sources: Vec<BlockSource>,
    pub status: Status,
    /// Tensors that took the place of a base tensor with the same output name.
    pub replaced: Vec<TensorDelta>,
    /// Aggregate over `replaced`; `None` when nothing was replaced.
    pub delta: Option<Delta>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockSource {
    pub alias: String,
    /// `None`: the tensor comes from outside any block of its source.
    pub index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Status {
    /// Every tensor from base, same name, same index.
    Unchanged,
    /// Entirely from one base block at another index (after a remove).
    Shifted { from: usize },
    /// Every tensor from one block of one donor.
    Replaced { alias: String, from: usize },
    /// A mix: `replaced` of `total` tensors are not the base's own.
    PartiallyReplaced { replaced: usize, total: usize },
    /// No block at this index in base.
    Added,
    /// Contains generated tensors (adapters).
    Generated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorDelta {
    pub name: String,
    pub delta: Delta,
}

/// `||new - old||_2 / ||old||_2` over every element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Delta {
    /// Weights were not read (`plan` without `--weights-diff`).
    NotComputed,
    /// Could not be measured; never reported as 0.
    Unknown {
        reason: String,
    },
    Relative {
        value: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRef {
    pub stack: String,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outside {
    pub name: String,
    pub origin: Origin,
    /// From base under the same name.
    pub unchanged: bool,
    /// `Some` when a base tensor of the same name was replaced.
    pub delta: Option<Delta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Origin {
    Source { alias: String, name: String },
    Generated,
}

impl From<&Payload> for Origin {
    fn from(p: &Payload) -> Self {
        match p {
            Payload::Source { alias, name } => Origin::Source { alias: alias.clone(), name: name.clone() },
            Payload::Generated(_) => Origin::Generated,
        }
    }
}

impl Report {
    /// The report of a compacted `plan` against `base`, the model of source `alias`.
    /// Weight deltas start as `NotComputed`; `measure` fills them.
    pub fn new(alias: &str, base: &Model, plan: &Plan) -> Self {
        let own = |name: &str, p: &Payload| matches!(p, Payload::Source { alias: a, name: n } if a == alias && n == name);
        let base_layout = base.layout();
        let out_layout = plan.to_model().layout();
        let base_blocks: BTreeSet<(&str, usize)> =
            base_layout.blocks.iter().map(|b| (b.stack.as_str(), b.index)).collect();
        let replacement = |name: &str| {
            let p = &plan.tensors[name].payload;
            (base.tensors.contains_key(name) && !own(name, p))
                .then(|| TensorDelta { name: name.into(), delta: Delta::NotComputed })
        };

        let mut stacks: BTreeMap<&str, Stack> = BTreeMap::new();
        for b in base_layout.blocks.iter().chain(&out_layout.blocks) {
            stacks.entry(&b.stack).or_insert_with(|| Stack {
                name: b.stack.clone(),
                blocks_before: base_layout.blocks.iter().filter(|x| x.stack == b.stack).count(),
                blocks_after: out_layout.blocks.iter().filter(|x| x.stack == b.stack).count(),
                blocks: Vec::new(),
            });
        }
        for b in &out_layout.blocks {
            let entries: Vec<(&str, &Payload)> =
                b.tensors.iter().map(|n| (n.as_str(), &plan.tensors[n].payload)).collect();
            let sources: BTreeSet<BlockSource> = entries
                .iter()
                .filter_map(|(_, p)| match p {
                    Payload::Source { alias, name } => {
                        Some(BlockSource { alias: alias.clone(), index: layout::block_of(name).map(|x| x.1) })
                    }
                    Payload::Generated(_) => None,
                })
                .collect();
            let foreign = entries.iter().filter(|(n, p)| !own(n, p)).count();
            let status = if entries.iter().any(|(_, p)| matches!(p, Payload::Generated(_))) {
                Status::Generated
            } else if !base_blocks.contains(&(b.stack.as_str(), b.index)) {
                Status::Added
            } else if foreign == 0 {
                Status::Unchanged
            } else {
                match sources.iter().collect::<Vec<_>>()[..] {
                    [BlockSource { alias: a, index: Some(from) }] if a == alias => {
                        Status::Shifted { from: *from }
                    }
                    [BlockSource { alias: a, index: Some(from) }] => {
                        Status::Replaced { alias: a.clone(), from: *from }
                    }
                    _ => Status::PartiallyReplaced { replaced: foreign, total: entries.len() },
                }
            };
            let replaced: Vec<TensorDelta> = b.tensors.iter().filter_map(|n| replacement(n)).collect();
            let delta = (!replaced.is_empty()).then_some(Delta::NotComputed);
            let block = BlockReport {
                index: b.index,
                tensors: entries.len(),
                sources: sources.into_iter().collect(),
                status,
                replaced,
                delta,
            };
            stacks.get_mut(b.stack.as_str()).expect("every output stack inserted above").blocks.push(block);
        }

        // A base tensor reaches the output if some planned tensor is read from it.
        let used: BTreeSet<&str> = plan
            .tensors
            .values()
            .filter_map(|p| match &p.payload {
                Payload::Source { alias: a, name } if a == alias => Some(name.as_str()),
                _ => None,
            })
            .collect();
        // Unused and not replaced in place: a donor tensor under the same name is a replacement.
        let gone = |n: &&String| {
            !used.contains(n.as_str())
                && plan
                    .tensors
                    .get(*n)
                    .is_none_or(|p| matches!(&p.payload, Payload::Source { alias: a, .. } if a == alias))
        };
        let mut removed_blocks = Vec::new();
        let mut removed_tensors = Vec::new();
        for b in &base_layout.blocks {
            let replaced_in_place = stacks[b.stack.as_str()].blocks.iter().any(|x| {
                x.index == b.index && !matches!(x.status, Status::Unchanged | Status::Shifted { .. })
            });
            if !replaced_in_place && b.tensors.iter().all(|n| !used.contains(n.as_str())) {
                removed_blocks.push(BlockRef { stack: b.stack.clone(), index: b.index });
            } else {
                removed_tensors.extend(b.tensors.iter().filter(gone).cloned());
            }
        }
        removed_tensors.extend(base_layout.other.iter().filter(gone).cloned());

        let outside = out_layout
            .other
            .iter()
            .map(|n| {
                let p = &plan.tensors[n].payload;
                Outside {
                    name: n.clone(),
                    origin: p.into(),
                    unchanged: own(n, p),
                    delta: replacement(n).map(|t| t.delta),
                }
            })
            .collect();

        Self {
            schema_version: SCHEMA_VERSION,
            base: alias.into(),
            stacks: stacks.into_values().collect(),
            removed_blocks,
            removed_tensors,
            outside,
            config_present: plan.companion.files.iter().any(|f| f == CONFIG),
            config_changes: plan.companion.changes.clone(),
            warnings: crate::build::warnings(plan),
            experimental: plan.experimental(),
            file: FileCheck::NotWritten,
            model_behavior: NOT_EVALUATED.into(),
        }
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self).expect("plain struct")
    }

    /// `out` -> `out.report.json`, `out.report.md`, next to the lock file.
    pub fn paths_for(output: &Path) -> [PathBuf; 2] {
        [".report.json", ".report.md"].map(|ext| {
            let mut p = output.as_os_str().to_owned();
            p.push(ext);
            p.into()
        })
    }

    pub fn save(&self, output: &Path) -> Result<[PathBuf; 2]> {
        let paths = Self::paths_for(output);
        for (path, text) in paths.iter().zip([self.json(), self.markdown()]) {
            fs::write(path, text).map_err(Error::io(path))?;
        }
        Ok(paths)
    }
}

#[cfg(test)]
#[path = "tests/report.rs"]
mod tests;
