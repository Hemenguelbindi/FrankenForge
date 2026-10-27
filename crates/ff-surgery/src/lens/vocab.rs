//! Token id -> readable text, from the model's tokenizer.json. Byte-level BPE (GPT-2, SmolLM2,
//! Llama 3) stores UTF-8 bytes as stand-in characters (`Ġ` for a space, `Ð¾` for `о`); those
//! are decoded back so a Russian or Chinese token reads as itself.

use std::path::Path;

use ff_core::{Error, Result};
use serde_json::Value;

use crate::compare::tokenizer;

/// Real vocabularies are below a million entries; an id past this is a damaged file.
const MAX_ID: u64 = 4_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vocab {
    texts: Vec<String>,
}

impl Vocab {
    /// `None` when the model folder has no tokenizer.json.
    pub fn load(model: &Path) -> Result<Option<Self>> {
        let Some((path, bytes)) = tokenizer::read(model)? else { return Ok(None) };
        let map = tokenizer::vocab(&path, &bytes)?;
        let json: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Format { path: path.clone(), reason: e.to_string() })?;
        let byte_level = is_byte_level(&json["decoder"]);
        let mut texts = Vec::new();
        for (token, id) in map {
            if id > MAX_ID {
                return Err(Error::Format { path, reason: format!("token id {id} is past {MAX_ID}") });
            }
            let id = id as usize;
            if texts.len() <= id {
                texts.resize(id + 1, String::new());
            }
            texts[id] = if byte_level { from_byte_level(&token) } else { token.replace('▁', " ") };
        }
        Ok(Some(Self { texts }))
    }

    pub fn from_texts(texts: Vec<String>) -> Self {
        Self { texts }
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Ids past the vocabulary (padding rows of a table) read as `<id>`.
    pub fn text(&self, id: usize) -> String {
        self.texts.get(id).filter(|t| !t.is_empty()).cloned().unwrap_or_else(|| format!("<{id}>"))
    }

    /// The token for a typed word: the word-start form (` word`) first, then `word` as is.
    pub fn find(&self, word: &str) -> Option<usize> {
        let word = word.trim();
        let spaced = format!(" {word}");
        self.texts.iter().position(|t| *t == spaced).or_else(|| self.texts.iter().position(|t| t == word))
    }
}

impl Vocab {
    /// Up to `limit` tokens that start the typed word (shortest first), for a word the model
    /// splits into several tokens.
    pub fn pieces_of(&self, word: &str, limit: usize) -> Vec<(usize, String)> {
        let word = word.trim().to_lowercase();
        let mut found: Vec<(usize, String)> = self
            .texts
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                let t = t.trim_start().to_lowercase();
                t.chars().count() > 1 && word.starts_with(&t)
            })
            .map(|(id, t)| (id, t.clone()))
            .collect();
        found.sort_by_key(|(_, t)| std::cmp::Reverse(t.trim().chars().count()));
        found.truncate(limit);
        found
    }
}

fn is_byte_level(decoder: &Value) -> bool {
    decoder["type"] == "ByteLevel"
        || decoder["decoders"].as_array().is_some_and(|d| d.iter().any(|x| x["type"] == "ByteLevel"))
}

/// Inverse of GPT-2's byte -> char table: printable bytes stand for themselves, the other 68
/// bytes were moved to 256.. in order.
fn byte_of(c: char) -> Option<u8> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let code = c as u32;
    if printable(code) {
        return Some(code as u8);
    }
    let n = code.checked_sub(256)?;
    (0u32..=255).filter(|b| !printable(*b)).nth(n as usize).map(|b| b as u8)
}

fn from_byte_level(token: &str) -> String {
    match token.chars().map(byte_of).collect::<Option<Vec<u8>>>() {
        Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        None => token.to_owned(),
    }
}

#[cfg(test)]
#[path = "tests/vocab.rs"]
mod tests;
