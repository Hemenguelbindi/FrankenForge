//! Name rules. Matching is on whole dot-separated segments, never on substrings, so
//! `q_proj_extra` is not a query projection and `enormous` is not a norm.

use super::{Param, Part, Role};

/// Parent modules whose children are attention projections.
const ATTENTION: &[&str] = &["self_attn", "attn", "attention", "cross_attn"];
/// Parent modules whose children are feed-forward (or MoE) matrices.
const FEED_FORWARD: &[&str] = &["mlp", "feed_forward", "ffn", "block_sparse_moe"];
const PROJECTORS: &[&str] = &["multi_modal_projector", "mm_projector", "embed_vision", "embed_audio"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub role: Role,
    pub param: Param,
    pub expert: Option<u64>,
}

fn mlp_part(segment: &str) -> Option<Part> {
    match segment {
        // `w1`/`w3`/`w2` is the Mixtral spelling of gate/up/down.
        "gate_proj" | "w1" => Some(Part::Gate),
        "up_proj" | "w3" => Some(Part::Up),
        "down_proj" | "w2" => Some(Part::Down),
        "gate_up_proj" => Some(Part::GateUp),
        _ => None,
    }
}

/// Plain decimal index (`7`, not `07` or `+7`), as written by every known exporter.
fn index(segment: &str) -> Option<u64> {
    let canonical = !segment.is_empty()
        && segment.bytes().all(|b| b.is_ascii_digit())
        && (segment == "0" || !segment.starts_with('0'));
    canonical.then(|| segment.parse().ok()).flatten()
}

fn is_norm(segment: &str) -> bool {
    matches!(segment, "norm" | "layernorm" | "layer_norm" | "ln_f" | "ln_1" | "ln_2" | "final_layernorm")
        || segment.ends_with("_layernorm")
        || segment.ends_with("_norm")
        || segment.starts_with("norm_")
}

pub fn classify(name: &str) -> Match {
    let mut segs: Vec<&str> = name.split('.').collect();
    let param = match segs.last() {
        Some(&"weight") => Param::Weight,
        Some(&"bias") => Param::Bias,
        _ => Param::Other,
    };
    if param != Param::Other {
        segs.pop();
    }
    // A wrapped linear layer (`q_proj.linear.weight`) is the same projection.
    if param != Param::Other && segs.last() == Some(&"linear") {
        segs.pop();
    }
    let found = |role, expert| Match { role, param, expert };
    let Some(&last) = segs.last() else { return found(Role::Unknown, None) };

    if param == Param::Other {
        let role = match last {
            "input_min" | "input_max" | "output_min" | "output_max" => Role::ClipBound,
            "position_embedding_table" => Role::PositionEmbedding,
            _ => Role::Unknown,
        };
        return found(role, None);
    }

    // `...experts.<n>.<part>`: one routed expert.
    if segs.len() >= 3
        && segs[segs.len() - 3] == "experts"
        && let Some(n) = index(segs[segs.len() - 2])
        && let Some(part) = mlp_part(last)
    {
        return found(Role::Expert(part), Some(n));
    }

    let parent = segs.len().checked_sub(2).map(|i| segs[i]);
    if parent == Some("shared_expert")
        && let Some(part) = mlp_part(last)
    {
        return found(Role::SharedExpert(part), None);
    }
    if let Some(p) = parent
        && ATTENTION.contains(&p)
    {
        let role = match last {
            "q_proj" => Some(Role::AttnQ),
            "k_proj" => Some(Role::AttnK),
            "v_proj" => Some(Role::AttnV),
            "o_proj" | "out_proj" => Some(Role::AttnO),
            "qkv_proj" | "query_key_value" | "c_attn" => Some(Role::AttnQkv),
            "q_norm" => Some(Role::QNorm),
            "k_norm" => Some(Role::KNorm),
            _ => None,
        };
        if let Some(role) = role {
            return found(role, None);
        }
    }
    if let Some(p) = parent
        && FEED_FORWARD.contains(&p)
    {
        let role = match last {
            "gate" | "router" => Some(Role::MoeRouter),
            "shared_expert_gate" => Some(Role::SharedExpertGate),
            other => mlp_part(other).map(Role::Mlp),
        };
        if let Some(role) = role {
            return found(role, None);
        }
    }

    let role = match last {
        "embed_tokens" | "wte" | "word_embeddings" => Role::TokenEmbedding,
        "embed_positions" | "wpe" | "position_embeddings" => Role::PositionEmbedding,
        "lm_head" => Role::OutputHead,
        _ if segs.iter().any(|s| PROJECTORS.contains(s)) => Role::Projector,
        // A norm outside every block, named plainly, normalizes the final hidden state.
        "norm" | "ln_f" | "final_layernorm" if ff_core::layout::block_of(name).is_none() => Role::FinalNorm,
        s if is_norm(s) => Role::Norm,
        _ => Role::Unknown,
    };
    found(role, None)
}

/// Architectures whose exact tensor paths are written out below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Llama,
    Qwen2,
    Qwen3,
    Qwen2Moe,
    Qwen3Moe,
    Mixtral,
    Gemma,
    Gemma2,
    Gemma3,
}

pub fn family(model_type: &str) -> Option<Family> {
    Some(match model_type {
        "llama" | "mistral" => Family::Llama,
        "qwen2" => Family::Qwen2,
        "qwen3" => Family::Qwen3,
        "qwen2_moe" => Family::Qwen2Moe,
        "qwen3_moe" => Family::Qwen3Moe,
        "mixtral" => Family::Mixtral,
        "gemma" => Family::Gemma,
        "gemma2" => Family::Gemma2,
        "gemma3_text" => Family::Gemma3,
        _ => return None,
    })
}

/// True only for the exact path the Hugging Face implementation of `family` uses.
pub fn is_canonical(family: Family, name: &str) -> bool {
    if matches!(name, "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight") {
        return true;
    }
    let Some((n, rel)) = name.strip_prefix("model.layers.").and_then(|r| r.split_once('.')) else {
        return false;
    };
    if index(n).is_none() {
        return false;
    }
    let base = matches!(
        rel,
        "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "input_layernorm.weight"
            | "post_attention_layernorm.weight"
    );
    let dense = || matches!(rel, "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight");
    let qkv_bias =
        || matches!(rel, "self_attn.q_proj.bias" | "self_attn.k_proj.bias" | "self_attn.v_proj.bias");
    let qk_norm = || matches!(rel, "self_attn.q_norm.weight" | "self_attn.k_norm.weight");
    let ffn_norms =
        || matches!(rel, "pre_feedforward_layernorm.weight" | "post_feedforward_layernorm.weight");
    let experts = |prefix: &str, parts: &[&str]| {
        rel.strip_prefix(prefix)
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(e, p)| index(e).is_some() && parts.iter().any(|x| p == format!("{x}.weight")))
    };
    let qwen_moe =
        || rel == "mlp.gate.weight" || experts("mlp.experts.", &["gate_proj", "up_proj", "down_proj"]);
    let shared = || {
        matches!(
            rel,
            "mlp.shared_expert.gate_proj.weight"
                | "mlp.shared_expert.up_proj.weight"
                | "mlp.shared_expert.down_proj.weight"
                | "mlp.shared_expert_gate.weight"
        )
    };
    base || match family {
        Family::Llama | Family::Gemma => dense(),
        Family::Qwen2 => dense() || qkv_bias(),
        Family::Qwen3 => dense() || qk_norm(),
        Family::Qwen2Moe => dense() || qkv_bias() || qwen_moe() || shared(),
        Family::Qwen3Moe => dense() || qk_norm() || qwen_moe(),
        Family::Mixtral => {
            rel == "block_sparse_moe.gate.weight" || experts("block_sparse_moe.experts.", &["w1", "w2", "w3"])
        }
        Family::Gemma2 => dense() || ffn_norms(),
        Family::Gemma3 => dense() || ffn_norms() || qk_norm(),
    }
}
