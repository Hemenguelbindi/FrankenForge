//! The safetensors header: `u64` LE length, then a JSON map of
//! `name -> {dtype, shape, data_offsets}` plus optional `__metadata__`.

use std::collections::HashMap;

use ff_core::{DType, Model, TensorInfo};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Header lengths above this are rejected before allocating.
pub const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;
const METADATA_KEY: &str = "__metadata__";

#[derive(Serialize, Deserialize)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub struct Parsed {
    pub model: Model,
    /// Byte range of each tensor, relative to the start of the data section.
    pub offsets: HashMap<String, (u64, u64)>,
}

/// Parses the JSON header and checks every tensor fits inside `data_len` bytes.
pub fn parse(json: &[u8], data_len: u64) -> Result<Parsed, String> {
    let map: Map<String, Value> = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let mut model = Model::default();
    let mut offsets = HashMap::new();
    for (name, value) in map {
        if name == METADATA_KEY {
            model.metadata = serde_json::from_value(value).map_err(|e| format!("{METADATA_KEY}: {e}"))?;
            continue;
        }
        let e: Entry = serde_json::from_value(value).map_err(|err| format!("{name}: {err}"))?;
        let [start, end] = e.data_offsets;
        let expected = e
            .shape
            .iter()
            .try_fold(e.dtype.size_bytes() as u64, |acc, &d| acc.checked_mul(d as u64))
            .ok_or_else(|| format!("{name}: size overflow"))?;
        if start > end || end > data_len || end - start != expected {
            return Err(format!(
                "{name}: offsets {start}..{end} do not fit {:?} {:?} in a {data_len}-byte data section",
                e.dtype, e.shape
            ));
        }
        offsets.insert(name.clone(), (start, end));
        model.insert(TensorInfo { name, dtype: e.dtype, shape: e.shape });
    }
    Ok(Parsed { model, offsets })
}

/// Length-prefixed header for `model`, tensors laid out back to back in map order.
pub fn serialize(model: &Model) -> Vec<u8> {
    let mut map = Map::new();
    if !model.metadata.is_empty() {
        map.insert(METADATA_KEY.into(), serde_json::to_value(&model.metadata).expect("string map"));
    }
    let mut offset = 0u64;
    for t in model.tensors.values() {
        let end = offset + t.byte_len() as u64;
        let entry = Entry { dtype: t.dtype, shape: t.shape.clone(), data_offsets: [offset, end] };
        map.insert(t.name.clone(), serde_json::to_value(entry).expect("plain struct"));
        offset = end;
    }
    let mut json = serde_json::to_vec(&map).expect("json map");
    // Pad so the data section starts 8-byte aligned, as the spec recommends.
    while !(8 + json.len()).is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend(json);
    out
}

#[cfg(test)]
#[path = "tests/header.rs"]
mod tests;
