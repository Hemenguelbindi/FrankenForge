//! Tests for module `attention`: KV size from config.json, including sliding windows, global
//! layers with their own head size, shared-KV layers, and Unknown for incomplete configs.

use serde_json::json;

use super::*;

#[test]
fn a_plain_llama_config_allocates_every_layer() {
    let c = json!({
        "hidden_size": 576, "intermediate_size": 1536, "num_attention_heads": 9,
        "num_key_value_heads": 3, "num_hidden_layers": 30, "vocab_size": 49152
    });
    let a = Attention::from_config(&c).get().cloned().unwrap();
    assert_eq!(a.caches.len(), 30);
    assert_eq!(a.caches[0], LayerCache { kv_heads: 3, head_dim: 64, window: None });
    assert_eq!(a.kv_cache_bytes(2048, 1, 2), 30 * 2 * 3 * 64 * 2048 * 2);
}

#[test]
fn windows_global_layers_and_shared_kv_follow_the_text_config() {
    let c = json!({ "text_config": {
        "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "num_hidden_layers": 5, "vocab_size": 100, "head_dim": 8, "global_head_dim": 16,
        "sliding_window": 4, "num_kv_shared_layers": 2,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention"]
    }});
    let a = Attention::from_config(&c).get().cloned().unwrap();
    // The last two layers reuse earlier caches: only three allocate.
    assert_eq!(a.caches.len(), 3);
    let ctx = 10;
    let sliding = 2 * 8 * 4; // window caps the positions at 4
    let full = 2 * 16 * ctx;
    assert_eq!(a.kv_cache_bytes(ctx, 1, 1), sliding + full + sliding);
}

#[test]
fn incomplete_or_inconsistent_configs_are_unknown() {
    let missing = json!({ "hidden_size": 8, "num_attention_heads": 2 });
    assert!(
        matches!(Attention::from_config(&missing), Known::Unknown(why) if why.contains("num_hidden_layers"))
    );
    let wrong = json!({
        "hidden_size": 8, "intermediate_size": 8, "num_attention_heads": 2, "num_hidden_layers": 3,
        "vocab_size": 10, "layer_types": ["full_attention"]
    });
    assert!(matches!(Attention::from_config(&wrong), Known::Unknown(why) if why.contains("layer_types")));
    let odd = json!({
        "hidden_size": 8, "intermediate_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1,
        "vocab_size": 10, "layer_types": ["mystery_attention"]
    });
    assert!(matches!(Attention::from_config(&odd), Known::Unknown(why) if why.contains("mystery_attention")));
}
