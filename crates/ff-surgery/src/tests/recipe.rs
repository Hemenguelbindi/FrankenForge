//! Tests for module `recipe`.

use super::*;

const EXAMPLE: &str = r#"
    name = "demo"
    base = "a"
    [sources]
    a = "a.safetensors"
    b = "b"
    [[steps]]
    op = "transplant"
    from = "b"
    src = "model.layers.3.mlp."
    dst = "model.layers.1.mlp."
    [[steps]]
    op = "adapter"
    name = "adapter.proj.weight"
    in_dim = 4
    out_dim = 2
"#;

#[test]
fn parses_steps_with_defaults() {
    let r = Recipe::parse(EXAMPLE).unwrap();
    assert_eq!(r.steps.len(), 2);
    assert!(matches!(r.steps[1], Step::Adapter { init: AdapterInit::Identity, dtype: DType::F32, .. }));
}

#[test]
fn rejects_typo_in_step() {
    let bad = EXAMPLE.replace("dst = ", "dest = ");
    assert!(Recipe::parse(&bad).is_err());
}
