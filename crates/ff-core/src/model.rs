use std::collections::BTreeMap;

use crate::{Error, Layout, Result, TensorInfo, layout};

/// Tensor-level view of a model: names, shapes and dtypes, no weights.
#[derive(Debug, Clone, Default)]
pub struct Model {
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

impl Model {
    pub fn insert(&mut self, info: TensorInfo) {
        self.tensors.insert(info.name.clone(), info);
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::TensorNotFound(name.to_owned()))
    }

    /// Tensors picked by `selector`; see [`selects`].
    pub fn select<'a>(&'a self, selector: &'a str) -> impl Iterator<Item = &'a TensorInfo> + 'a {
        self.tensors.values().filter(move |t| selects(selector, &t.name))
    }

    pub fn param_count(&self) -> usize {
        self.tensors.values().map(TensorInfo::numel).sum()
    }

    pub fn layout(&self) -> Layout {
        layout::detect(self.tensors.keys().map(String::as_str))
    }
}

/// A selector ending in `.` is a block prefix (`model.layers.3.`), anything
/// else is one exact tensor name. This keeps `layers.1` from matching `layers.10`.
pub fn selects(selector: &str, name: &str) -> bool {
    if selector.ends_with('.') { name.starts_with(selector) } else { name == selector }
}

#[cfg(test)]
#[path = "tests/model.rs"]
mod tests;
