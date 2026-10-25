//! The shape numbers an estimate needs, read from a model's `config.json` (its `text_config` when
//! there is one). Missing fields make the shape Unknown instead of being guessed from the model
//! name.

use serde_json::Value;

use crate::known::Known;

/// The KV cache one layer allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerCache {
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Sliding-window layers keep at most this many positions.
    pub window: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attention {
    pub hidden: u64,
    pub intermediate: u64,
    pub heads: u64,
    pub vocab: u64,
    pub layers: u64,
    /// One entry per layer that stores its own KV cache; layers sharing another layer's cache
    /// (`num_kv_shared_layers`) are not listed.
    pub caches: Vec<LayerCache>,
}

fn int(c: &Value, key: &str) -> Option<u64> {
    c.get(key).and_then(Value::as_u64)
}

impl Attention {
    pub fn from_config(config: &Value) -> Known<Attention> {
        let c = config.get("text_config").unwrap_or(config);
        let need = |key: &str| int(c, key).ok_or_else(|| format!("config.json has no `{key}`"));
        let parsed = (|| -> Result<Attention, String> {
            let (hidden, heads, layers) =
                (need("hidden_size")?, need("num_attention_heads")?, need("num_hidden_layers")?);
            let intermediate = need("intermediate_size")?;
            let vocab = need("vocab_size")?;
            let kv_heads = int(c, "num_key_value_heads").unwrap_or(heads);
            let head_dim = int(c, "head_dim").unwrap_or(hidden / heads.max(1));
            let window = int(c, "sliding_window");
            let types: Vec<String> = match c.get("layer_types").and_then(Value::as_array) {
                Some(t) => t.iter().map(|v| v.as_str().unwrap_or_default().to_owned()).collect(),
                None => vec!["full_attention".into(); layers as usize],
            };
            if types.len() as u64 != layers {
                return Err(format!("layer_types has {} entries for {layers} layers", types.len()));
            }
            let shared = int(c, "num_kv_shared_layers").unwrap_or(0).min(layers);
            let caches = types[..(layers - shared) as usize]
                .iter()
                .map(|t| match t.as_str() {
                    "sliding_attention" => Ok(LayerCache { kv_heads, head_dim, window }),
                    "full_attention" => Ok(LayerCache {
                        kv_heads: int(c, "num_global_key_value_heads").unwrap_or(kv_heads),
                        head_dim: int(c, "global_head_dim").unwrap_or(head_dim),
                        window: None,
                    }),
                    other => Err(format!("unknown layer type `{other}`")),
                })
                .collect::<Result<_, _>>()?;
            Ok(Attention { hidden, intermediate, heads, vocab, layers, caches })
        })();
        match parsed {
            Ok(a) => Known::Known(a),
            Err(why) => Known::Unknown(why),
        }
    }

    /// K and V for `context` positions of `batch` sequences, `bytes` per element.
    pub fn kv_cache_bytes(&self, context: u64, batch: u64, bytes: u64) -> u64 {
        self.caches
            .iter()
            .map(|l| {
                2 * l.kv_heads * l.head_dim * l.window.map_or(context, |w| w.min(context)) * batch * bytes
            })
            .sum()
    }
}

#[cfg(test)]
#[path = "tests/attention.rs"]
mod tests;
