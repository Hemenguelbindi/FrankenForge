//! Tests for module `layout`.

use super::*;

#[test]
fn groups_blocks_and_reports_gaps() {
    let layout = detect([
        "model.embed_tokens.weight",
        "model.layers.0.mlp.weight",
        "model.layers.0.attn.weight",
        "model.layers.2.mlp.weight",
        "lm_head.weight",
    ]);
    assert_eq!(layout.blocks.len(), 2);
    assert_eq!(layout.blocks[0].prefix, "model.layers.0.");
    assert_eq!(layout.blocks[0].tensors.len(), 2);
    assert_eq!(layout.other, ["model.embed_tokens.weight", "lm_head.weight"]);
    assert_eq!(layout.gaps().len(), 1);
}

#[test]
fn gpt2_style_names() {
    assert_eq!(
        block_of("transformer.h.11.attn.c_attn.weight"),
        Some(("transformer.h".into(), 11, "transformer.h.11.".into()))
    );
}
