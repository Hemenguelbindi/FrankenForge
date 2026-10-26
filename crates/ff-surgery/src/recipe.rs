use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{DType, Error, Result};
use serde::{Deserialize, Serialize};

/// A reproducible build description, stored as TOML.
/// Source paths are relative to the recipe file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub name: String,
    /// Alias of the source whose tensors are the starting point.
    pub base: String,
    pub sources: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub steps: Vec<Step>,
}

/// One edit. Selectors ending in `.` match a block prefix, otherwise one tensor name.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Step {
    /// Copy tensors under `src` of source `from` to `dst`, replacing same-named ones.
    Transplant {
        from: String,
        src: String,
        dst: String,
        /// Keep tensors under `dst` that the donor has no counterpart for (a partial replacement);
        /// without it they make the step fail instead of silently staying.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        keep_extra: bool,
    },
    /// Drop tensors under `select`.
    Remove { select: String },
    /// Move tensors under `src` to `dst`, e.g. to close layer-index gaps.
    Rename { src: String, dst: String },
    /// Add an explicit `[out_dim, in_dim]` projection bridging two hidden sizes.
    Adapter {
        name: String,
        in_dim: usize,
        out_dim: usize,
        #[serde(default)]
        init: AdapterInit,
        #[serde(default = "default_adapter_dtype")]
        dtype: DType,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterInit {
    /// Identity on the overlapping square, zeros elsewhere.
    #[default]
    Identity,
    Zeros,
}

fn default_adapter_dtype() -> DType {
    DType::F32
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(Error::io(path))?;
        Self::parse(&text).map_err(|e| Error::Recipe(format!("{}: {e}", path.display())))
    }

    pub fn parse(text: &str) -> std::result::Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// The recipe as TOML that `parse` reads back unchanged; fails on a non-UTF-8 source path.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Recipe(format!("cannot write recipe: {e}")))
    }

    /// Directory that relative source paths resolve against.
    pub fn base_dir(recipe_path: &Path) -> &Path {
        recipe_path.parent().unwrap_or(Path::new(""))
    }
}

#[cfg(test)]
#[path = "tests/recipe.rs"]
mod tests;
