use std::collections::BTreeMap;

use ff_core::{Model, TensorInfo, layout, selects};

use serde::{Deserialize, Serialize};

use crate::AdapterInit;
use crate::companion::Companion;

/// What was not validated in a build with adapters. Written to `plan` output and the lock file so
/// the result is never presented as a ready working model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Experimental {
    pub experimental: bool,
    /// No runtime executes the added adapters.
    pub runtime_support: String,
    pub numerical_validation: String,
    pub quality_validation: String,
    /// Generated tensors that make the build experimental.
    pub tensors: Vec<String>,
}

impl Experimental {
    pub fn text(&self) -> String {
        format!(
            "experimental: {}\nruntime_support: {}\nnumerical_validation: {}\nquality_validation: {}\ngenerated tensors: {}\n",
            self.experimental,
            self.runtime_support,
            self.numerical_validation,
            self.quality_validation,
            self.tensors.join(", ")
        )
    }
}

/// Where an output tensor's bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    Source { alias: String, name: String },
    Generated(AdapterInit),
}

#[derive(Debug, Clone)]
pub struct Planned {
    pub info: TensorInfo,
    pub payload: Payload,
}

/// The output model as metadata plus provenance; no weights are loaded.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub tensors: BTreeMap<String, Planned>,
    pub metadata: BTreeMap<String, String>,
    /// config.json and tokenizer of the base source; filled in by `build::plan`.
    pub companion: Companion,
}

impl Plan {
    pub fn from_base(alias: &str, model: &Model) -> Self {
        let tensors = model
            .tensors
            .values()
            .map(|t| {
                let payload = Payload::Source { alias: alias.into(), name: t.name.clone() };
                (t.name.clone(), Planned { info: t.clone(), payload })
            })
            .collect();
        Self { tensors, metadata: model.metadata.clone(), ..Self::default() }
    }

    /// Renumbers every stack consecutively from 0, keeping block order, so that
    /// remove/transplant leave no gaps. The byte source (`Payload`) is unchanged.
    pub fn compact(&mut self) {
        let mut next: BTreeMap<String, usize> = BTreeMap::new();
        let mut moved = Vec::new();
        // Take every block out first: a new name may equal the old name of another block.
        for block in self.to_model().layout().blocks {
            let index = next.entry(block.stack.clone()).or_default();
            let prefix = format!("{}.{index}.", block.stack);
            *index += 1;
            for name in &block.tensors {
                if let Some(mut p) = self.tensors.remove(name) {
                    p.info.name = format!("{prefix}{}", &name[block.prefix.len()..]);
                    moved.push(p);
                }
            }
        }
        for p in moved {
            self.insert(p);
        }
    }

    pub fn insert(&mut self, p: Planned) {
        self.tensors.insert(p.info.name.clone(), p);
    }

    pub fn select(&self, selector: &str) -> Vec<String> {
        self.tensors.keys().filter(|n| selects(selector, n)).cloned().collect()
    }

    pub fn to_model(&self) -> Model {
        let mut m = Model { metadata: self.metadata.clone(), ..Model::default() };
        for p in self.tensors.values() {
            m.insert(p.info.clone());
        }
        m
    }

    /// Whether the plan changes block composition, order, indices or origin relative to base:
    /// a base block was removed, or a stack tensor does not come from base under the same name.
    pub fn blocks_changed(&self, alias: &str, base: &Model) -> bool {
        let in_block = |name: &str| layout::block_of(name).is_some();
        let removed = base.tensors.keys().any(|n| in_block(n) && !self.tensors.contains_key(n));
        removed
            || self.tensors.values().any(|p| {
                in_block(&p.info.name)
                    && !matches!(&p.payload, Payload::Source { alias: a, name } if a == alias && *name == p.info.name)
            })
    }

    /// `Some` if the plan contains generated adapters.
    pub fn experimental(&self) -> Option<Experimental> {
        let tensors: Vec<String> = self
            .tensors
            .values()
            .filter(|p| matches!(p.payload, Payload::Generated(_)))
            .map(|p| p.info.name.clone())
            .collect();
        (!tensors.is_empty()).then(|| Experimental {
            experimental: true,
            runtime_support: "missing".into(),
            numerical_validation: "not_performed".into(),
            quality_validation: "not_performed".into(),
            tensors,
        })
    }

    /// Tensor count per origin: a source alias or `<generated>`.
    pub fn provenance(&self) -> BTreeMap<&str, usize> {
        let mut out = BTreeMap::new();
        for p in self.tensors.values() {
            let key = match &p.payload {
                Payload::Source { alias, .. } => alias.as_str(),
                Payload::Generated(_) => "<generated>",
            };
            *out.entry(key).or_default() += 1;
        }
        out
    }
}

#[cfg(test)]
#[path = "tests/plan.rs"]
mod tests;
