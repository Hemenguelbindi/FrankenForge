//! Files next to the weights: config.json, tokenizer, chat templates. Taken from the base
//! source directory; config.json is adjusted to the plan, everything else is copied byte for byte.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, Layout, Result};
use serde_json::Value;

use crate::lock::hash_path;

/// Layer-count keys of common architectures (Llama/Qwen, GPT-2, T5).
const LAYER_KEYS: &[&str] = &["num_hidden_layers", "n_layer", "num_layers"];
pub const CONFIG: &str = "config.json";
/// Keys that tie a layer's structure to its index (which layers are MoE, which use a
/// sliding window). Removing blocks shifts indices, so the key then describes other layers:
/// in Qwen-MoE with `decoder_sparse_step = 2`, removing dense layer 0 puts an MoE block in its place.
const INDEX_PATTERN_KEYS: &[&str] = &[
    "decoder_sparse_step",
    "mlp_only_layers",
    "max_window_layers",
    "sliding_window_pattern",
    "first_k_dense_replace",
    "moe_layer_freq",
    "interleave_moe_layer_step",
];

#[derive(Debug, Clone, Default)]
pub struct Companion {
    /// Base source directory; `None` if base is a single file.
    pub dir: Option<PathBuf>,
    /// Files copied unchanged (including config.json when it was not edited).
    pub files: Vec<String>,
    /// Edited config.json; `None` if nothing needs to change.
    pub config: Option<Value>,
    /// What changed in config.json, shown by `plan`.
    pub changes: Vec<String>,
}

/// Tokenizer and model descriptions. Weights and the shard index are not copied: the output is written anew.
pub fn is_companion(name: &str) -> bool {
    matches!(
        name,
        CONFIG
            | "generation_config.json"
            | "special_tokens_map.json"
            | "added_tokens.json"
            | "merges.txt"
            | "preprocessor_config.json"
            | "processor_config.json"
    ) || name.starts_with("tokenizer")
        || name.starts_with("vocab.")
        || name.starts_with("chat_template")
}

impl Companion {
    /// Reads directory `source`; `before`/`after` are the block layouts of the source model and the plan,
    /// `blocks_changed` means the plan changes block composition, order, indices or origin.
    pub fn load(source: &Path, before: &Layout, after: &Layout, blocks_changed: bool) -> Result<Self> {
        if !source.is_dir() {
            return Ok(Self::default());
        }
        let mut files = Vec::new();
        for entry in fs::read_dir(source).map_err(Error::io(source))? {
            let path = entry.map_err(Error::io(source))?.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && path.is_file()
                && is_companion(name)
            {
                files.push(name.to_owned());
            }
        }
        files.sort();
        let mut c = Self { dir: Some(source.to_owned()), files, ..Self::default() };

        // Blocks untouched: the config is still correct and is copied byte for byte.
        if blocks_changed && c.files.iter().any(|f| f == CONFIG) {
            let path = source.join(CONFIG);
            let unsupported = |why: String| Error::UnsupportedConfig(format!("{}: {why}", path.display()));
            let mut config = read_json(&path)?;
            index_pattern(&config).map_err(unsupported)?;
            let (before, after) = (stack_sizes(before), stack_sizes(after));
            if before != after {
                c.changes = patch(&mut config, &before, &after).map_err(unsupported)?;
                c.files.retain(|f| f != CONFIG);
                c.config = Some(config);
            }
        }
        Ok(c)
    }

    /// Writes the files to `out`, re-reads them and checks them against the source / plan.
    pub fn write(&self, out: &Path) -> Result<()> {
        let Some(src) = &self.dir else { return Ok(()) };
        for name in &self.files {
            let (from, to) = (src.join(name), out.join(name));
            fs::copy(&from, &to).map_err(Error::io(&to))?;
            if hash_path(&from)? != hash_path(&to)? {
                return Err(Error::Verify(format!("{name}: copy differs from source")));
            }
        }
        if let Some(config) = &self.config {
            let path = out.join(CONFIG);
            let json = serde_json::to_string_pretty(config).expect("json value");
            fs::write(&path, json).map_err(Error::io(&path))?;
            if &read_json(&path)? != config {
                return Err(Error::Verify(format!("{CONFIG}: re-read differs from plan")));
            }
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(Error::io(path))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Format { path: path.to_owned(), reason: e.to_string() })
}

/// A config with an index-dependent key allows only a no-op: any block change (composition,
/// order, indices, origin) can put a layer of the wrong kind at its index.
fn index_pattern(config: &Value) -> std::result::Result<(), String> {
    let set = |v: &Value| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty());
    match INDEX_PATTERN_KEYS.iter().find(|k| config.get(**k).is_some_and(set)) {
        Some(k) => {
            Err(format!("`{k}` ties layer structure to layer index; only a no-op export is supported"))
        }
        None => Ok(()),
    }
}

/// Number of blocks in each stack.
fn stack_sizes(layout: &Layout) -> BTreeMap<String, u64> {
    let mut sizes = BTreeMap::new();
    for b in &layout.blocks {
        *sizes.entry(b.stack.clone()).or_default() += 1;
    }
    sizes
}

/// Updates the layer count when that can be done completely; otherwise returns the reason to refuse.
/// Every changed stack must map unambiguously to one key from `LAYER_KEYS`.
// shortcut: per-layer arrays and nested `*_config` are refused, not re-sliced; add when a model needs it.
fn patch(
    config: &mut Value,
    before: &BTreeMap<String, u64>,
    after: &BTreeMap<String, u64>,
) -> std::result::Result<Vec<String>, String> {
    let obj = config.as_object_mut().ok_or("not a JSON object")?;
    if let Some((k, _)) =
        obj.iter().find(|(_, v)| v.as_object().is_some_and(|o| LAYER_KEYS.iter().any(|l| o.contains_key(*l))))
    {
        return Err(format!("nested `{k}` has its own layer count; updating it is not supported"));
    }

    let mut changes = Vec::new();
    let mut covered = Vec::new();
    for key in LAYER_KEYS {
        let Some(old) = obj.get(*key).and_then(Value::as_u64) else { continue };
        let stacks: Vec<&String> = before.iter().filter(|&(_, &n)| n == old).map(|(s, _)| s).collect();
        let [stack] = stacks[..] else {
            return Err(format!(
                "`{key}` = {old} matches {} layer stacks, expected exactly one",
                stacks.len()
            ));
        };
        let new = after.get(stack).copied().unwrap_or(0);
        if new != old {
            if let Some((k, _)) =
                obj.iter().find(|(_, v)| v.as_array().is_some_and(|a| a.len() as u64 == old))
            {
                return Err(format!("per-layer `{k}` has {old} entries; re-slicing it is not supported"));
            }
            changes.push(format!("{CONFIG}: {key} {old} -> {new}"));
        }
        covered.push((key, stack.clone(), new));
    }

    let changed = before.keys().chain(after.keys()).filter(|s| before.get(*s) != after.get(*s));
    for stack in changed {
        if !covered.iter().any(|(_, s, _)| s == stack) {
            return Err(format!("layer stack `{stack}` changed but no layer count in config describes it"));
        }
    }
    for (key, _, new) in covered {
        obj.insert((*key).into(), new.into());
    }
    Ok(changes)
}

#[cfg(test)]
#[path = "tests/companion.rs"]
mod tests;
