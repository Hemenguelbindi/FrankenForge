//! Plain-language model map: the same components and blocks as the technical tree, but every
//! group and part is named by what it does. Raw names, shapes, dtypes and numbers stay in the
//! Technical presentation.

use eframe::egui::{self, RichText};

use crate::components::{Component, Kind};
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::semantics::{Group, Part, Role};
use crate::views;

/// Left panel: model facts and the components found in the file. No dtypes, config or raw names.
pub(crate) fn overview(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    selected: &mut Option<Kind>,
    picked: &mut Option<String>,
) {
    ui.label(RichText::new(t.t("structure-title")).small().weak());
    ui.heading(&inspected.name);
    ui.label(RichText::new(inspected.path.display().to_string()).small().weak());
    ui.add_space(8.0);

    views::stat(ui, t.t("overview-parameters"), t.count(inspected.params));
    let blocks: usize = inspected.components.iter().map(Component::block_count).sum();
    views::stat(ui, t.t("structure-blocks"), blocks.to_string());
    ui.add_space(12.0);

    ui.label(RichText::new(t.t("structure-parts")).strong());
    for c in &inspected.components {
        let size = if c.block_count() > 0 {
            t.ta("card-blocks", &[("blocks", c.block_count().into()), ("params", t.count(c.params).into())])
        } else {
            t.ta("card-tensors", &[("tensors", c.loose.len().into()), ("params", t.count(c.params).into())])
        };
        let label = format!("{} · {size}", t.t(c.kind.message_keys().0));
        if ui.selectable_label(*selected == Some(c.kind), label).clicked() {
            if *selected != Some(c.kind) {
                *picked = None;
            }
            *selected = Some(c.kind);
        }
    }
    ui.add_space(12.0);
    ui.label(RichText::new(t.t("structure-no-raw")).small().weak());
}

pub(crate) fn group_keys(group: Group) -> (&'static str, &'static str) {
    match group {
        Group::Attention => ("structure-group-attention", "structure-group-attention-desc"),
        Group::Mlp => ("structure-group-mlp", "structure-group-mlp-desc"),
        Group::Experts => ("structure-group-experts", "structure-group-experts-desc"),
        Group::Normalization => ("structure-group-normalization", "structure-group-normalization-desc"),
        Group::Other => ("structure-group-other", "structure-group-other-desc"),
    }
}

/// Human title and description for every tensor role. No Q/K/V, projection or other shorthand:
/// those appear only in Technical data.
pub(crate) fn role_keys(role: Role) -> (&'static str, &'static str) {
    use Part::*;
    match role {
        Role::TokenEmbedding => ("structure-role-token-embedding", "structure-role-token-embedding-desc"),
        Role::PositionEmbedding => {
            ("structure-role-position-embedding", "structure-role-position-embedding-desc")
        }
        Role::AttnQ => ("structure-role-attn-q", "structure-role-attn-q-desc"),
        Role::AttnK => ("structure-role-attn-k", "structure-role-attn-k-desc"),
        Role::AttnV => ("structure-role-attn-v", "structure-role-attn-v-desc"),
        Role::AttnO => ("structure-role-attn-o", "structure-role-attn-o-desc"),
        Role::AttnQkv => ("structure-role-attn-qkv", "structure-role-attn-qkv-desc"),
        Role::QNorm => ("structure-role-q-norm", "structure-role-q-norm-desc"),
        Role::KNorm => ("structure-role-k-norm", "structure-role-k-norm-desc"),
        Role::Mlp(Gate) => ("structure-role-mlp-gate", "structure-role-mlp-gate-desc"),
        Role::Mlp(Up) => ("structure-role-mlp-up", "structure-role-mlp-up-desc"),
        Role::Mlp(Down) => ("structure-role-mlp-down", "structure-role-mlp-down-desc"),
        Role::Mlp(GateUp) => ("structure-role-mlp-gate-up", "structure-role-mlp-gate-up-desc"),
        Role::MoeRouter => ("structure-role-router", "structure-role-router-desc"),
        Role::Expert(Gate) => ("structure-role-expert-gate", "structure-role-expert-desc"),
        Role::Expert(Up) => ("structure-role-expert-up", "structure-role-expert-desc"),
        Role::Expert(Down) => ("structure-role-expert-down", "structure-role-expert-desc"),
        Role::Expert(GateUp) => ("structure-role-expert-gate-up", "structure-role-expert-desc"),
        Role::SharedExpert(Gate) => ("structure-role-shared-gate", "structure-role-shared-desc"),
        Role::SharedExpert(Up) => ("structure-role-shared-up", "structure-role-shared-desc"),
        Role::SharedExpert(Down) => ("structure-role-shared-down", "structure-role-shared-desc"),
        Role::SharedExpert(GateUp) => ("structure-role-shared-gate-up", "structure-role-shared-desc"),
        Role::SharedExpertGate => ("structure-role-shared-router", "structure-role-shared-router-desc"),
        Role::Norm => ("structure-role-norm", "structure-role-norm-desc"),
        Role::FinalNorm => ("structure-role-final-norm", "structure-role-final-norm-desc"),
        Role::OutputHead => ("structure-role-output", "structure-role-output-desc"),
        Role::Projector => ("structure-role-projector", "structure-role-projector-desc"),
        Role::ClipBound => ("structure-role-clip", "structure-role-clip-desc"),
        Role::Unknown => ("structure-role-unknown", "structure-role-unknown-desc"),
    }
}

#[cfg(test)]
#[path = "tests/structure.rs"]
mod tests;
