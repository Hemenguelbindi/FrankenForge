//! Meaning of each shape dimension. A meaning is shown only when the role implies it; config.json
//! then either confirms the size (basis of the role) or contradicts it (meaning dropped, Unknown).

use serde_json::Value;

use super::{Basis, Param, Part, Role};

#[derive(Debug, Clone, PartialEq)]
pub struct Meaning {
    pub key: &'static str,
    /// The config values behind the size, e.g. `8 × 256`, when it is a product.
    pub formula: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dim {
    pub size: usize,
    pub meaning: Option<Meaning>,
    pub basis: Basis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Hidden,
    Vocab,
    HeadsXHeadDim,
    KvHeadsXHeadDim,
    HeadDim,
    Intermediate,
    MoeIntermediate,
    SharedIntermediate,
    Experts,
}

/// Linear weights are `[out_features, in_features]` (the PyTorch convention all exporters keep).
fn expected(role: Role) -> &'static [Expect] {
    use Expect::*;
    match role {
        Role::TokenEmbedding | Role::OutputHead => &[Vocab, Hidden],
        Role::AttnQ => &[HeadsXHeadDim, Hidden],
        Role::AttnK | Role::AttnV => &[KvHeadsXHeadDim, Hidden],
        Role::AttnO => &[Hidden, HeadsXHeadDim],
        Role::QNorm | Role::KNorm => &[HeadDim],
        Role::Mlp(Part::Gate | Part::Up) => &[Intermediate, Hidden],
        Role::Mlp(Part::Down) => &[Hidden, Intermediate],
        Role::Expert(Part::Gate | Part::Up) => &[MoeIntermediate, Hidden],
        Role::Expert(Part::Down) => &[Hidden, MoeIntermediate],
        Role::SharedExpert(Part::Gate | Part::Up) => &[SharedIntermediate, Hidden],
        Role::SharedExpert(Part::Down) => &[Hidden, SharedIntermediate],
        Role::MoeRouter => &[Experts, Hidden],
        Role::Norm | Role::FinalNorm => &[Hidden],
        _ => &[],
    }
}

#[cfg(test)]
pub const KEYS: [&str; 9] = [
    key(Expect::Hidden),
    key(Expect::Vocab),
    key(Expect::HeadsXHeadDim),
    key(Expect::KvHeadsXHeadDim),
    key(Expect::HeadDim),
    key(Expect::Intermediate),
    key(Expect::MoeIntermediate),
    key(Expect::SharedIntermediate),
    key(Expect::Experts),
];

const fn key(e: Expect) -> &'static str {
    match e {
        Expect::Hidden => "dim-hidden",
        Expect::Vocab => "dim-vocab",
        Expect::HeadsXHeadDim => "dim-heads-x-head-dim",
        Expect::KvHeadsXHeadDim => "dim-kv-heads-x-head-dim",
        Expect::HeadDim => "dim-head-dim",
        Expect::Intermediate => "dim-intermediate",
        Expect::MoeIntermediate => "dim-moe-intermediate",
        Expect::SharedIntermediate => "dim-shared-intermediate",
        Expect::Experts => "dim-experts",
    }
}

fn int(section: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| section.get(*k).and_then(Value::as_u64))
}

/// The size config.json implies, and the formula when it is a product.
fn from_config(e: Expect, c: &Value) -> Option<(u64, Option<String>)> {
    let hidden = int(c, &["hidden_size", "n_embd", "d_model"]);
    let heads = int(c, &["num_attention_heads", "n_head"]);
    let head_dim = int(c, &["head_dim"]).or_else(|| hidden?.checked_div(heads?));
    let product = |a: Option<u64>, b: Option<u64>| {
        let (a, b) = (a?, b?);
        Some((a.checked_mul(b)?, Some(format!("{a} × {b}"))))
    };
    match e {
        Expect::Hidden => hidden.map(|v| (v, None)),
        Expect::Vocab => int(c, &["vocab_size"]).map(|v| (v, None)),
        Expect::HeadsXHeadDim => product(heads, head_dim),
        Expect::KvHeadsXHeadDim => product(int(c, &["num_key_value_heads"]).or(heads), head_dim),
        Expect::HeadDim => head_dim.map(|v| (v, None)),
        Expect::Intermediate => int(c, &["intermediate_size", "n_inner"]).map(|v| (v, None)),
        // Mixtral keeps the expert width in `intermediate_size`.
        Expect::MoeIntermediate => int(c, &["moe_intermediate_size", "intermediate_size"]).map(|v| (v, None)),
        Expect::SharedIntermediate => int(c, &["shared_expert_intermediate_size"]).map(|v| (v, None)),
        Expect::Experts => int(c, &["num_experts", "num_local_experts"]).map(|v| (v, None)),
    }
}

pub fn explain(role: Role, param: Param, shape: &[usize], config: Option<&Value>, basis: Basis) -> Vec<Dim> {
    let mut expect = expected(role);
    // A bias has one entry per output feature, i.e. the weight's first dimension.
    if param == Param::Bias {
        expect = &expect[..expect.len().min(1)];
    }
    let unknown = |size| Dim { size, meaning: None, basis: Basis::Unknown };
    if param == Param::Other || expect.len() != shape.len() {
        return shape.iter().map(|&s| unknown(s)).collect();
    }
    shape
        .iter()
        .zip(expect)
        .map(|(&size, &e)| match config.map(|c| from_config(e, c)) {
            // Config confirms the size: the meaning stands on the role's own basis.
            Some(Some((v, formula))) if v == size as u64 => {
                Dim { size, meaning: Some(Meaning { key: key(e), formula }), basis }
            }
            // Config says something else: our expectation does not hold for this model.
            Some(Some(_)) => unknown(size),
            // Nothing to check against: the meaning follows from the role alone.
            _ => Dim { size, meaning: Some(Meaning { key: key(e), formula: None }), basis: Basis::Inferred },
        })
        .collect()
}
