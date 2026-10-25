//! What a tensor is for, derived from its name, shape and config.json only, never from its
//! weights. Every statement carries a basis, so a guess is never shown as a fact:
//! `Confirmed` needs a known architecture declared in config.json and the exact tensor path of
//! that architecture; a name-only match is `Inferred`; anything else is `Unknown`.

mod dims;
mod rules;

use ff_core::TensorInfo;
use serde_json::Value;

pub use dims::Dim;
#[cfg(test)]
pub use dims::KEYS as DIM_KEYS;
pub use rules::classify;

use crate::components::{Kind, modality};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Basis {
    Confirmed,
    Inferred,
    Unknown,
}

impl Basis {
    /// (label, explanation) message keys.
    pub fn message_keys(self) -> (&'static str, &'static str) {
        match self {
            Basis::Confirmed => ("basis-confirmed", "basis-confirmed-hint"),
            Basis::Inferred => ("basis-inferred", "basis-inferred-hint"),
            Basis::Unknown => ("basis-unknown", "basis-unknown-hint"),
        }
    }
}

/// One of the matrices of a gated MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Gate,
    Up,
    Down,
    GateUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    TokenEmbedding,
    PositionEmbedding,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    AttnQkv,
    QNorm,
    KNorm,
    Mlp(Part),
    MoeRouter,
    Expert(Part),
    SharedExpert(Part),
    SharedExpertGate,
    Norm,
    FinalNorm,
    OutputHead,
    Projector,
    ClipBound,
    Unknown,
}

/// How the tensors of one block are grouped in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Attention,
    Mlp,
    Experts,
    Normalization,
    Other,
}

impl Group {
    #[cfg(test)]
    pub const ALL: [Group; 5] =
        [Group::Attention, Group::Mlp, Group::Experts, Group::Normalization, Group::Other];

    pub fn message_key(self) -> &'static str {
        match self {
            Group::Attention => "group-attention",
            Group::Mlp => "group-mlp",
            Group::Experts => "group-experts",
            Group::Normalization => "group-normalization",
            Group::Other => "group-other",
        }
    }
}

impl Role {
    pub fn group(self) -> Group {
        match self {
            Role::AttnQ | Role::AttnK | Role::AttnV | Role::AttnO | Role::AttnQkv => Group::Attention,
            Role::Mlp(_) => Group::Mlp,
            Role::MoeRouter | Role::Expert(_) | Role::SharedExpert(_) | Role::SharedExpertGate => {
                Group::Experts
            }
            Role::QNorm | Role::KNorm | Role::Norm | Role::FinalNorm => Group::Normalization,
            _ => Group::Other,
        }
    }

    /// (title, description) message keys. Expert titles take an `$index` argument.
    pub fn message_keys(self) -> (&'static str, &'static str) {
        use Part::*;
        match self {
            Role::TokenEmbedding => ("role-token-embedding", "role-token-embedding-desc"),
            Role::PositionEmbedding => ("role-position-embedding", "role-position-embedding-desc"),
            Role::AttnQ => ("role-attn-q", "role-attn-q-desc"),
            Role::AttnK => ("role-attn-k", "role-attn-k-desc"),
            Role::AttnV => ("role-attn-v", "role-attn-v-desc"),
            Role::AttnO => ("role-attn-o", "role-attn-o-desc"),
            Role::AttnQkv => ("role-attn-qkv", "role-attn-qkv-desc"),
            Role::QNorm => ("role-q-norm", "role-q-norm-desc"),
            Role::KNorm => ("role-k-norm", "role-k-norm-desc"),
            Role::Mlp(Gate) => ("role-mlp-gate", "role-mlp-gate-desc"),
            Role::Mlp(Up) => ("role-mlp-up", "role-mlp-up-desc"),
            Role::Mlp(Down) => ("role-mlp-down", "role-mlp-down-desc"),
            Role::Mlp(GateUp) => ("role-mlp-gate-up", "role-mlp-gate-up-desc"),
            Role::MoeRouter => ("role-moe-router", "role-moe-router-desc"),
            Role::Expert(Gate) => ("role-expert-gate", "role-expert-desc"),
            Role::Expert(Up) => ("role-expert-up", "role-expert-desc"),
            Role::Expert(Down) => ("role-expert-down", "role-expert-desc"),
            Role::Expert(GateUp) => ("role-expert-gate-up", "role-expert-desc"),
            Role::SharedExpert(Gate) => ("role-shared-expert-gate", "role-shared-expert-desc"),
            Role::SharedExpert(Up) => ("role-shared-expert-up", "role-shared-expert-desc"),
            Role::SharedExpert(Down) => ("role-shared-expert-down", "role-shared-expert-desc"),
            Role::SharedExpert(GateUp) => ("role-shared-expert-gate-up", "role-shared-expert-desc"),
            Role::SharedExpertGate => ("role-shared-expert-router", "role-shared-expert-router-desc"),
            Role::Norm => ("role-norm", "role-norm-desc"),
            Role::FinalNorm => ("role-final-norm", "role-final-norm-desc"),
            Role::OutputHead => ("role-output-head", "role-output-head-desc"),
            Role::Projector => ("role-projector", "role-projector-desc"),
            Role::ClipBound => ("role-clip-bound", "role-clip-bound-desc"),
            Role::Unknown => ("role-unknown", "role-unknown-desc"),
        }
    }

    #[cfg(test)]
    pub const ALL: [Role; 29] = {
        use Part::*;
        [
            Role::TokenEmbedding,
            Role::PositionEmbedding,
            Role::AttnQ,
            Role::AttnK,
            Role::AttnV,
            Role::AttnO,
            Role::AttnQkv,
            Role::QNorm,
            Role::KNorm,
            Role::Mlp(Gate),
            Role::Mlp(Up),
            Role::Mlp(Down),
            Role::Mlp(GateUp),
            Role::MoeRouter,
            Role::Expert(Gate),
            Role::Expert(Up),
            Role::Expert(Down),
            Role::Expert(GateUp),
            Role::SharedExpert(Gate),
            Role::SharedExpert(Up),
            Role::SharedExpert(Down),
            Role::SharedExpert(GateUp),
            Role::SharedExpertGate,
            Role::Norm,
            Role::FinalNorm,
            Role::OutputHead,
            Role::Projector,
            Role::ClipBound,
            Role::Unknown,
        ]
    };
}

/// Weight, bias, or another named parameter (a scalar bound, a scale).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    Weight,
    Bias,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorDoc {
    pub role: Role,
    pub basis: Basis,
    pub param: Param,
    pub expert: Option<u64>,
    /// The `model_type` whose exact naming confirmed the role, for the basis explanation.
    pub architecture: Option<String>,
    pub dims: Vec<Dim>,
}

/// config.json as the semantics see it.
#[derive(Debug, Clone, Copy)]
pub struct ConfigRef<'a> {
    pub value: &'a Value,
    /// Read from the checkpoint directory itself. A config found next to a single file may
    /// belong to another model: it can check sizes, never confirm an architecture.
    pub own: bool,
}

/// A config without an expert count cannot rule an index out.
fn expert_in_range(section: Option<&Value>, index: u64) -> bool {
    let count = section
        .and_then(|s| s.get("num_experts").or_else(|| s.get("num_local_experts")))
        .and_then(Value::as_u64);
    count.is_none_or(|count| index < count)
}

/// The section of config.json matching the tensor's component is used.
pub fn describe(info: &TensorInfo, config: Option<ConfigRef<'_>>) -> TensorDoc {
    let m = classify(&info.name);
    let modal = modality(&info.name);
    let section = config.map(|c| {
        let key = match modal {
            Some(Kind::Audio) => "audio_config",
            Some(Kind::Vision) => "vision_config",
            _ => "text_config",
        };
        c.value.get(key).filter(|s| s.is_object()).unwrap_or(c.value)
    });
    // Exact architecture rules exist only for the language part of known models.
    let architecture = section
        .filter(|_| modal.is_none() && config.is_some_and(|c| c.own))
        .and_then(|s| s.get("model_type"))
        .and_then(Value::as_str)
        .filter(|t| rules::family(t).is_some_and(|f| rules::is_canonical(f, &info.name)))
        // A routed expert beyond the declared count cannot be that architecture's exact path.
        .filter(|_| m.expert.is_none_or(|n| expert_in_range(section, n)));
    let mut basis = match (m.role, architecture) {
        (Role::Unknown, _) => Basis::Unknown,
        (_, Some(_)) => Basis::Confirmed,
        (_, None) => Basis::Inferred,
    };
    let mut dims = dims::explain(m.role, m.param, &info.shape, section, basis);
    // A config that contradicts the shape does not describe this tensor: the name alone remains.
    if basis == Basis::Confirmed && dims.iter().any(|d| d.basis == Basis::Unknown) {
        basis = Basis::Inferred;
        dims = dims::explain(m.role, m.param, &info.shape, section, basis);
    }
    TensorDoc {
        role: m.role,
        basis,
        param: m.param,
        expert: m.expert,
        architecture: architecture.filter(|_| basis == Basis::Confirmed).map(str::to_owned),
        dims,
    }
}

#[cfg(test)]
#[path = "tests/semantics.rs"]
mod tests;
