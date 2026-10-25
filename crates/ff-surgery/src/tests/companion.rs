//! Tests for module `companion`.

use super::*;
use ff_core::layout::detect;
use serde_json::json;

/// Source model: 3 blocks in `m.layers` and 1 in `v.blocks`; after the plan `m.layers` has 2.
fn sizes() -> (BTreeMap<String, u64>, BTreeMap<String, u64>) {
    let before = detect(["m.layers.0.w", "m.layers.1.w", "m.layers.2.w", "v.blocks.0.w"]);
    let after = detect(["m.layers.0.w", "m.layers.1.w", "v.blocks.0.w"]);
    (stack_sizes(&before), stack_sizes(&after))
}

fn refusal(mut config: Value) -> String {
    let (before, after) = sizes();
    patch(&mut config, &before, &after).unwrap_err()
}

#[test]
fn patches_layer_count_of_matching_stack() {
    let (before, after) = sizes();
    let mut config = json!({ "num_hidden_layers": 3, "hidden_size": 4, "architectures": ["X"] });
    let changes = patch(&mut config, &before, &after).unwrap();
    assert_eq!(config["num_hidden_layers"], 2);
    assert_eq!(config["hidden_size"], 4);
    assert_eq!(changes, ["config.json: num_hidden_layers 3 -> 2"]);
}

#[test]
fn refuses_per_layer_array() {
    let why = refusal(json!({ "num_hidden_layers": 3, "layer_types": ["a", "b", "c"] }));
    assert!(why.contains("layer_types"), "{why}");
}

#[test]
fn refuses_nested_text_config() {
    let why = refusal(json!({ "num_hidden_layers": 3, "text_config": { "num_hidden_layers": 3 } }));
    assert!(why.contains("text_config"), "{why}");
}

#[test]
fn refuses_index_dependent_layer_structure() {
    // qwen3-moe-tiny-random config: layer 0 is dense, layer 1 is MoE, decided by index.
    let why =
        index_pattern(&json!({ "num_hidden_layers": 2, "decoder_sparse_step": 2, "mlp_only_layers": [] }))
            .unwrap_err();
    assert!(why.contains("decoder_sparse_step"), "{why}");
    assert!(index_pattern(&json!({ "mlp_only_layers": [0] })).unwrap_err().contains("mlp_only_layers"));
    // Empty and null values set nothing.
    assert!(index_pattern(&json!({ "mlp_only_layers": [], "sliding_window_pattern": null })).is_ok());
}

#[test]
fn refuses_ambiguous_or_unknown_stack() {
    // The key matches no stack.
    assert!(refusal(json!({ "num_hidden_layers": 3, "n_layer": 7 })).contains("matches 0"));
    // The changed stack is described by no key.
    assert!(refusal(json!({ "hidden_size": 4 })).contains("m.layers"));
    // Two stacks of the same size: unclear which one the key describes.
    let before = stack_sizes(&detect(["a.layers.0.w", "a.layers.1.w", "b.blocks.0.w", "b.blocks.1.w"]));
    let after = stack_sizes(&detect(["a.layers.0.w", "b.blocks.0.w", "b.blocks.1.w"]));
    let why = patch(&mut json!({ "num_hidden_layers": 2 }), &before, &after).unwrap_err();
    assert!(why.contains("matches 2"), "{why}");
}

#[test]
fn picks_tokenizer_files_not_weights() {
    for name in ["tokenizer.json", "tokenizer.model", "vocab.json", "chat_template.jinja", CONFIG] {
        assert!(is_companion(name), "{name}");
    }
    for name in ["model.safetensors", "model.safetensors.index.json", "README.md"] {
        assert!(!is_companion(name), "{name}");
    }
}
