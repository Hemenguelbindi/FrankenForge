//! Tests for module `semantics`: known names, unknown names, false matches and dimensions.

use super::*;
use ff_core::DType;
use serde_json::json;

fn info(name: &str, shape: &[usize]) -> TensorInfo {
    TensorInfo { name: name.into(), dtype: DType::BF16, shape: shape.to_vec() }
}

/// The config of qwen3-moe-tiny-random.
fn qwen3_moe() -> Value {
    json!({
        "model_type": "qwen3_moe", "hidden_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 32, "intermediate_size": 128, "moe_intermediate_size": 128, "num_experts": 8,
        "vocab_size": 151936
    })
}

fn meaning(d: &Dim) -> Option<(&str, Option<&str>)> {
    d.meaning.as_ref().map(|m| (m.key, m.formula.as_deref()))
}

/// config.json read from the checkpoint directory itself.
fn own(value: &Value) -> Option<ConfigRef<'_>> {
    Some(ConfigRef { value, own: true })
}

#[test]
fn config_next_to_a_single_file_checks_sizes_but_never_confirms() {
    let c = qwen3_moe();
    let sibling = Some(ConfigRef { value: &c, own: false });
    let q = describe(&info("model.layers.0.self_attn.q_proj.weight", &[64, 64]), sibling);
    assert_eq!((q.role, q.basis, q.architecture), (Role::AttnQ, Basis::Inferred, None));
    assert_eq!(meaning(&q.dims[0]), Some(("dim-heads-x-head-dim", Some("2 × 32"))));
    assert!(q.dims.iter().all(|d| d.basis == Basis::Inferred));
}

#[test]
fn a_contradicting_config_downgrades_a_confirmed_role() {
    // Canonical qwen3_moe name, but the shape does not fit the declared sizes.
    let d = describe(&info("model.layers.0.self_attn.q_proj.weight", &[128, 128]), own(&qwen3_moe()));
    assert_eq!((d.role, d.basis, d.architecture), (Role::AttnQ, Basis::Inferred, None));
    assert!(d.dims.iter().all(|x| x.basis == Basis::Unknown));
}

#[test]
fn mixtral_expert_width_comes_from_intermediate_size() {
    let c = json!({ "model_type": "mixtral", "hidden_size": 64, "intermediate_size": 224, "num_local_experts": 8 });
    let e = describe(&info("model.layers.0.block_sparse_moe.experts.3.w1.weight", &[224, 64]), own(&c));
    assert_eq!((e.role, e.basis), (Role::Expert(Part::Gate), Basis::Confirmed));
    assert!(e.dims.iter().all(|d| d.basis == Basis::Confirmed));
}

#[test]
fn known_architecture_paths_are_confirmed_with_checked_dimensions() {
    let c = qwen3_moe();
    let q = describe(&info("model.layers.0.self_attn.q_proj.weight", &[64, 64]), own(&c));
    assert_eq!(
        (q.role, q.basis, q.architecture.as_deref()),
        (Role::AttnQ, Basis::Confirmed, Some("qwen3_moe"))
    );
    assert_eq!(meaning(&q.dims[0]), Some(("dim-heads-x-head-dim", Some("2 × 32"))));
    assert_eq!(meaning(&q.dims[1]), Some(("dim-hidden", None)));
    assert!(q.dims.iter().all(|d| d.basis == Basis::Confirmed));

    let k = describe(&info("model.layers.0.self_attn.k_proj.weight", &[32, 64]), own(&c));
    assert_eq!(meaning(&k.dims[0]), Some(("dim-kv-heads-x-head-dim", Some("1 × 32"))));

    let e = describe(&info("model.layers.1.mlp.experts.3.down_proj.weight", &[64, 128]), own(&c));
    assert_eq!((e.role, e.expert, e.basis), (Role::Expert(Part::Down), Some(3), Basis::Confirmed));
    assert_eq!(e.role.group(), Group::Experts);
    assert_eq!(meaning(&e.dims[1]), Some(("dim-moe-intermediate", None)));

    let r = describe(&info("model.layers.1.mlp.gate.weight", &[8, 64]), own(&c));
    assert_eq!((r.role, r.basis), (Role::MoeRouter, Basis::Confirmed));
    assert_eq!(meaning(&r.dims[0]), Some(("dim-experts", None)));

    let n = describe(&info("model.layers.0.self_attn.q_norm.weight", &[32]), own(&c));
    assert_eq!((n.role, n.role.group(), n.basis), (Role::QNorm, Group::Normalization, Basis::Confirmed));

    let emb = describe(&info("model.embed_tokens.weight", &[151936, 64]), own(&c));
    assert_eq!((emb.role, emb.basis), (Role::TokenEmbedding, Basis::Confirmed));
    assert_eq!(meaning(&emb.dims[0]), Some(("dim-vocab", None)));
}

#[test]
fn same_names_without_a_known_architecture_are_only_inferred() {
    let name = "model.layers.0.self_attn.q_proj.weight";
    // No config at all: the meaning follows from the role, nothing is verified.
    let bare = describe(&info(name, &[64, 64]), None);
    assert_eq!((bare.role, bare.basis), (Role::AttnQ, Basis::Inferred));
    assert!(bare.dims.iter().all(|d| d.basis == Basis::Inferred && d.meaning.is_some()));
    // An architecture without exact rules here (gemma4): sizes check out, the basis stays Inferred.
    let gemma = json!({ "model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "hidden_size": 1536, "num_attention_heads": 8, "head_dim": 256 } });
    let g =
        describe(&info("model.language_model.layers.0.self_attn.q_proj.weight", &[2048, 1536]), own(&gemma));
    assert_eq!((g.role, g.basis, g.architecture), (Role::AttnQ, Basis::Inferred, None));
    assert_eq!(meaning(&g.dims[0]), Some(("dim-heads-x-head-dim", Some("8 × 256"))));
    assert_eq!(g.dims[0].basis, Basis::Inferred);
}

#[test]
fn an_expert_index_beyond_the_declared_count_is_only_inferred() {
    let c = qwen3_moe();
    let last = describe(&info("model.layers.0.mlp.experts.7.down_proj.weight", &[64, 128]), own(&c));
    assert_eq!(last.basis, Basis::Confirmed, "index 7 is the last declared expert");
    let beyond = describe(&info("model.layers.0.mlp.experts.8.down_proj.weight", &[64, 128]), own(&c));
    assert_eq!(
        (beyond.role, beyond.expert, beyond.basis),
        (Role::Expert(Part::Down), Some(8), Basis::Inferred)
    );
}

#[test]
fn substrings_and_odd_paths_are_not_mistaken_for_known_parts() {
    let c = qwen3_moe();
    let role = |name: &str| describe(&info(name, &[4]), own(&c));
    for name in [
        "model.layers.0.self_attn.q_proj_extra.weight",
        "model.enormous.weight",
        "foo.k_proj.weight",
        "model.layers.0.mlp.experts.gate.weight",
        "model.layers.0.mlp.experts.007.up_proj.weight",
        "model.layers.0.attn_q_proj.weight",
    ] {
        let d = role(name);
        assert_eq!((d.role, d.basis), (Role::Unknown, Basis::Unknown), "{name}");
    }
    // Recognized by name, but not the exact path of the declared architecture: not Confirmed.
    for name in
        ["model.layers.07.self_attn.q_proj.weight", "model.vision_tower.layers.0.self_attn.q_proj.weight"]
    {
        let d = role(name);
        assert_eq!((d.role, d.basis), (Role::AttnQ, Basis::Inferred), "{name}");
    }
    // A dense model has no router, so a `gate` there is only a guess.
    let llama = json!({ "model_type": "llama" });
    let g = describe(&info("model.layers.0.mlp.gate.weight", &[8, 64]), own(&llama));
    assert_eq!((g.role, g.basis), (Role::MoeRouter, Basis::Inferred));
}

#[test]
fn dimensions_contradicted_by_config_are_unknown() {
    let d = describe(&info("model.layers.0.self_attn.q_proj.weight", &[65, 64]), own(&qwen3_moe()));
    assert_eq!((d.dims[0].meaning.as_ref(), d.dims[0].basis), (None, Basis::Unknown));
    assert_eq!(meaning(&d.dims[1]), Some(("dim-hidden", None)));
    // Wrong rank: nothing is claimed about any dimension.
    let r = describe(&info("model.layers.0.self_attn.q_proj.weight", &[64, 64, 1]), own(&qwen3_moe()));
    assert!(r.dims.iter().all(|d| d.meaning.is_none() && d.basis == Basis::Unknown));
}

#[test]
fn gemma_style_and_auxiliary_names() {
    let d = |name: &str| classify(name);
    let q = d("model.language_model.layers.0.self_attn.q_proj.linear.weight");
    assert_eq!((q.role, q.param), (Role::AttnQ, Param::Weight));
    assert_eq!(d("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max").role, Role::ClipBound);
    assert_eq!(d("model.embed_vision.embedding_projection.weight").role, Role::Projector);
    assert_eq!(d("model.language_model.norm.weight").role, Role::FinalNorm);
    assert_eq!(d("model.layers.0.post_feedforward_layernorm.weight").role, Role::Norm);
    assert_eq!(d("model.layers.0.block_sparse_moe.experts.2.w2.weight").role, Role::Expert(Part::Down));
    assert_eq!(d("model.layers.0.mlp.shared_expert.up_proj.weight").role, Role::SharedExpert(Part::Up));
    assert_eq!(d("lm_head.weight").role, Role::OutputHead);
}

#[test]
fn bias_is_confirmed_only_where_the_architecture_has_one() {
    let name = "model.layers.0.self_attn.q_proj.bias";
    let qwen2 = json!({ "model_type": "qwen2", "hidden_size": 64, "num_attention_heads": 2 });
    let b = describe(&info(name, &[64]), own(&qwen2));
    assert_eq!((b.role, b.param, b.basis), (Role::AttnQ, Param::Bias, Basis::Confirmed));
    assert_eq!(meaning(&b.dims[0]), Some(("dim-heads-x-head-dim", Some("2 × 32"))));
    // Qwen3 has no attention bias, so the same name is only inferred there.
    let qwen3 = json!({ "model_type": "qwen3" });
    assert_eq!(describe(&info(name, &[64]), own(&qwen3)).basis, Basis::Inferred);
}
