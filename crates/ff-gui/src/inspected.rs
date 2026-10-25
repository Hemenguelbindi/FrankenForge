//! What the inspector shows for one opened checkpoint. Built from `ff-core`/`ff-io` only,
//! so nothing here depends on egui and it can be unit-tested.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use std::sync::Arc;

use ff_core::{DType, Layout, Model, ModelFormat, Result, TensorSource};
use serde_json::Value;

use crate::components::{Component, components};
use crate::semantics::ConfigRef;

/// config.json is untrusted and shown as text every frame; real ones are a few KB.
const MAX_CONFIG_LEN: u64 = 1024 * 1024;

pub struct Inspected {
    pub path: PathBuf,
    /// Folder name: file names like `model.safetensors` say nothing about the model.
    pub name: String,
    pub model: Model,
    /// Lowercased tensor names in `model.tensors` order, so typing in search allocates nothing.
    lower_names: Vec<String>,
    pub layout: Layout,
    pub params: usize,
    pub dtypes: BTreeMap<DType, usize>,
    pub components: Vec<Component>,
    pub config: Config,
    /// Parsed config.json for tensor semantics; `None` unless `config` is `Json`.
    pub config_value: Option<Value>,
    /// True only for a checkpoint directory. A config.json next to a single file may belong to
    /// another model, so it can check sizes but never confirm an architecture.
    pub config_is_own: bool,
    /// The source the metadata came from; analysis reads through it, so the bytes always match
    /// the dtype and shape on screen and headers are not parsed again.
    pub source: Arc<dyn TensorSource>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Config {
    /// No config.json in the checkpoint's folder.
    Missing,
    /// Where it was read from, and the pretty-printed JSON.
    Json { path: PathBuf, text: String },
    /// config.json exists but is unusable; the weights still open. `detail` is the original
    /// error, shown as technical details next to the localized explanation of `problem`.
    Invalid { problem: ConfigProblem, detail: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigProblem {
    TooLarge,
    Unreadable,
    NotJson,
}

impl ConfigProblem {
    #[cfg(test)]
    pub const ALL: [ConfigProblem; 3] =
        [ConfigProblem::TooLarge, ConfigProblem::Unreadable, ConfigProblem::NotJson];

    pub fn message_key(self) -> &'static str {
        match self {
            ConfigProblem::TooLarge => "config-too-large",
            ConfigProblem::Unreadable => "config-unreadable",
            ConfigProblem::NotJson => "config-not-json",
        }
    }
}

impl Inspected {
    /// Reads headers only: `format.open` parses metadata and never loads tensor bytes.
    pub fn open(path: &Path, format: &dyn ModelFormat) -> Result<Self> {
        let source: Arc<dyn TensorSource> = Arc::from(format.open(path)?);
        let model = source.model().clone();
        let mut dtypes = BTreeMap::new();
        for t in model.tensors.values() {
            *dtypes.entry(t.dtype).or_default() += 1;
        }
        let layout = model.layout();
        let folder = folder(path);
        let (config, config_value) = folder.map_or((Config::Missing, None), read_config);
        let name = folder
            .and_then(Path::file_name)
            .or_else(|| path.file_name())
            .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        Ok(Self {
            path: path.to_owned(),
            name,
            lower_names: model.tensors.keys().map(|n| n.to_lowercase()).collect(),
            params: model.param_count(),
            dtypes,
            components: components(&model, &layout),
            config,
            config_value,
            config_is_own: path.is_dir(),
            source,
            layout,
            model,
        })
    }

    pub fn config_ref(&self) -> Option<ConfigRef<'_>> {
        self.config_value.as_ref().map(|value| ConfigRef { value, own: self.config_is_own })
    }

    /// Names of tensors containing `query`, ignoring case. A blank query returns nothing:
    /// the UI shows the tree instead, so there is no point copying every name.
    pub fn search(&self, query: &str) -> Vec<String> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        self.model
            .tensors
            .keys()
            .zip(&self.lower_names)
            .filter(|(_, lower)| lower.contains(&q))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// The checkpoint directory itself, or the folder holding a single checkpoint file.
fn folder(path: &Path) -> Option<&Path> {
    if path.is_dir() { Some(path) } else { path.parent() }
}

/// For display the config next to a single file is shown too (with its path); building still
/// takes config.json only from a checkpoint directory.
fn read_config(folder: &Path) -> (Config, Option<Value>) {
    let path = folder.join("config.json");
    if !path.is_file() {
        return (Config::Missing, None);
    }
    let invalid = |problem: ConfigProblem, reason: String| {
        (Config::Invalid { problem, detail: format!("{}: {reason}", path.display()) }, None)
    };
    // The limit applies to the read itself, so a file that grows after a size check cannot slip past it.
    let mut text = String::new();
    let read = fs::File::open(&path).and_then(|f| f.take(MAX_CONFIG_LEN + 1).read_to_string(&mut text));
    match read {
        Ok(n) if n as u64 > MAX_CONFIG_LEN => {
            return invalid(ConfigProblem::TooLarge, format!("larger than {MAX_CONFIG_LEN} bytes"));
        }
        Ok(_) => {}
        Err(e) => return invalid(ConfigProblem::Unreadable, e.to_string()),
    }
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => {
            let text = serde_json::to_string_pretty(&v).expect("a parsed JSON value serializes");
            (Config::Json { text, path }, Some(v))
        }
        Err(e) => invalid(ConfigProblem::NotJson, e.to_string()),
    }
}

#[cfg(test)]
#[path = "tests/inspected.rs"]
mod tests;
