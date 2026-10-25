//! Tests for module `model`.

use super::selects;

#[test]
fn prefix_needs_trailing_dot() {
    assert!(selects("model.layers.1.", "model.layers.1.mlp.weight"));
    assert!(!selects("model.layers.1.", "model.layers.10.mlp.weight"));
    assert!(!selects("model.layers.1", "model.layers.10.mlp.weight"));
    assert!(selects("lm_head.weight", "lm_head.weight"));
}
