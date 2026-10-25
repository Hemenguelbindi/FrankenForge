//! Compares tokenizer.json of two models: first byte for byte, then by the token -> id vocabulary.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, Result};
use serde::Serialize;
use serde_json::Value;

const FILE: &str = "tokenizer.json";
/// Untrusted file: never read more than this (real tokenizer.json files are tens of MB).
const MAX_LEN: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerMatch {
    /// Files are byte-identical.
    Identical,
    /// Files differ but the token -> id vocabulary (with added_tokens) is the same.
    SameVocab,
    DifferentVocab,
    /// At least one model has no tokenizer.json.
    Missing,
}

#[derive(Debug, Serialize)]
pub struct TokenizerDiff {
    pub status: TokenizerMatch,
    pub vocab_a: Option<usize>,
    pub vocab_b: Option<usize>,
}

/// `a`, `b` are model paths; tokenizer.json is looked up only inside a model directory.
// shortcut: only tokenizer.json is compared, not tokenizer.model / vocab.json; add when such a model shows up.
pub fn compare(a: &Path, b: &Path) -> Result<TokenizerDiff> {
    let (fa, fb) = (read(a)?, read(b)?);
    let (va, vb) = (vocab_of(&fa)?, vocab_of(&fb)?);
    let status = match (&fa, &fb) {
        (Some((_, x)), Some((_, y))) if x == y => TokenizerMatch::Identical,
        (Some(_), Some(_)) if va == vb => TokenizerMatch::SameVocab,
        (Some(_), Some(_)) => TokenizerMatch::DifferentVocab,
        _ => TokenizerMatch::Missing,
    };
    Ok(TokenizerDiff { status, vocab_a: va.map(|v| v.len()), vocab_b: vb.map(|v| v.len()) })
}

fn read(model: &Path) -> Result<Option<(PathBuf, Vec<u8>)>> {
    let path = model.join(FILE);
    if !model.is_dir() || !path.is_file() {
        return Ok(None);
    }
    let len = fs::metadata(&path).map_err(Error::io(&path))?.len();
    if len > MAX_LEN {
        return Err(Error::Format { path, reason: format!("{len} bytes, limit {MAX_LEN}") });
    }
    let bytes = fs::read(&path).map_err(Error::io(&path))?;
    Ok(Some((path, bytes)))
}

fn vocab_of(file: &Option<(PathBuf, Vec<u8>)>) -> Result<Option<BTreeMap<String, u64>>> {
    file.as_ref().map(|(path, bytes)| vocab(path, bytes)).transpose()
}

/// token -> id from `model.vocab` (an object for BPE/WordPiece, a `[token, score]` array for Unigram)
/// and `added_tokens`.
pub fn vocab(path: &Path, bytes: &[u8]) -> Result<BTreeMap<String, u64>> {
    let bad = |reason: String| Error::Format { path: path.to_owned(), reason };
    let json: Value = serde_json::from_slice(bytes).map_err(|e| bad(e.to_string()))?;
    let mut map = BTreeMap::new();
    match &json["model"]["vocab"] {
        Value::Object(o) => {
            for (token, id) in o {
                let id = id.as_u64().ok_or_else(|| bad(format!("id of `{token}` is not an integer")))?;
                map.insert(token.clone(), id);
            }
        }
        Value::Array(entries) => {
            for (id, e) in entries.iter().enumerate() {
                let token = e
                    .get(0)
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad("vocab entry is not [token, score]".into()))?;
                map.insert(token.to_owned(), id as u64);
            }
        }
        _ => return Err(bad("model.vocab is missing".into())),
    }
    for t in json["added_tokens"].as_array().into_iter().flatten() {
        if let (Some(content), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
            map.insert(content.to_owned(), id);
        }
    }
    Ok(map)
}

#[cfg(test)]
#[path = "tests/tokenizer.rs"]
mod tests;
