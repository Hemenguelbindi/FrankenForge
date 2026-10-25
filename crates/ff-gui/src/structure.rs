//! Plain-language model map: the same components and blocks as the technical tree, but every
//! group and part is named by what it does. Raw names, shapes, dtypes and numbers stay in the
//! Technical presentation.

use std::collections::BTreeMap;

use eframe::egui::{self, RichText};

use crate::components::{Component, Kind};
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::semantics::{Group, Param, Part, Role, classify};
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

/// Central panel for the chosen component. Clicking a part selects it; the summary near the top
/// offers the explicit jump to Technical data, where Q/K/V, raw names, shapes and numbers live.
pub(crate) fn tree(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    c: &Component,
    picked: &mut Option<String>,
    open_technical: &mut bool,
) {
    ui.heading(t.t(c.kind.message_keys().0));
    ui.label(t.t(c.kind.message_keys().1));
    ui.add_space(6.0);
    ui.label(RichText::new(t.t("structure-explainer")).weak());
    ui.add_space(6.0);

    // Keep the selection summary and its jump button above the scroll area, otherwise a long
    // block list would push them off the bottom of the panel.
    if let Some(name) = picked.as_deref()
        && belongs(c, inspected, name)
    {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(t.t("structure-selected")).weak());
            let m = classify(name);
            ui.label(RichText::new(t.t(role_keys(m.role).0)).strong());
            if let Some(index) = m.expert {
                ui.label(
                    RichText::new(t.ta("structure-expert-index", &[("index", index.into())])).small().weak(),
                );
            }
            if ui.button(t.t("structure-open-technical")).clicked() {
                *open_technical = true;
            }
        });
        ui.separator();
    }

    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        if c.stacks.is_empty() && c.loose.is_empty() {
            ui.label(t.t("structure-empty"));
            return;
        }
        for stack in &c.stacks {
            ui.label(
                RichText::new(t.ta("structure-stack", &[("count", stack.blocks.len().into())])).strong(),
            );
            for &i in &stack.blocks {
                let block = &inspected.layout.blocks[i];
                let title = t.ta("structure-block", &[("index", (block.index + 1).into())]);
                egui::CollapsingHeader::new(title)
                    .id_salt(("structure", &block.prefix))
                    .default_open(stack.blocks.len() == 1)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(
                                t.ta("structure-block-parts", &[("count", block.tensors.len().into())]),
                            )
                            .small()
                            .weak(),
                        );
                        let names: Vec<&String> = block.tensors.iter().collect();
                        grouped(ui, t, c.kind, &block.prefix, &names, picked);
                    });
            }
            ui.add_space(8.0);
        }
        if !c.loose.is_empty() {
            let title = t.ta("structure-loose", &[("count", c.loose.len().into())]);
            egui::CollapsingHeader::new(title)
                .id_salt(("structure", c.kind, "loose"))
                // Same guard as the technical tree: a flat checkpoint can hold 100k+ loose tensors.
                .default_open(c.stacks.is_empty() && c.loose.len() <= views::AUTO_OPEN_ROWS)
                .show(ui, |ui| {
                    let names: Vec<&String> = c.loose.iter().collect();
                    grouped(ui, t, c.kind, "loose", &names, picked);
                });
        }
    });
}

/// True when `name` is one of this component's tensors (inside one of its blocks or loose).
fn belongs(c: &Component, inspected: &Inspected, name: &str) -> bool {
    c.loose.iter().any(|n| n.as_str() == name)
        || c.stacks.iter().any(|s| {
            s.blocks.iter().any(|&i| inspected.layout.blocks[i].tensors.iter().any(|n| n.as_str() == name))
        })
}

fn grouped(
    ui: &mut egui::Ui,
    t: &I18n,
    kind: Kind,
    id: &str,
    names: &[&String],
    picked: &mut Option<String>,
) {
    let mut groups: BTreeMap<Group, Vec<&String>> = BTreeMap::new();
    for name in names {
        groups.entry(classify(name).role.group()).or_default().push(*name);
    }
    for (group, names) in groups {
        let (title, description) = group_keys(group);
        // Keep big lists closed by default (freeze guard, and a large expert group would bury
        // everything else); the `kind` salt stops one component's toggles bleeding into another.
        let open = names.len() <= views::AUTO_OPEN_ROWS && (group != Group::Experts || names.len() <= 8);
        egui::CollapsingHeader::new(t.t(title)).id_salt((kind, id, group)).default_open(open).show(
            ui,
            |ui| {
                ui.label(RichText::new(t.t(description)).small().weak());
                let mut seen = Vec::new();
                for name in names {
                    row(ui, t, name, picked, &mut seen);
                }
            },
        );
    }
}

/// One part inside a group. The description is shown once per role, so eight parallel units do
/// not repeat the same paragraph eight times.
fn row(ui: &mut egui::Ui, t: &I18n, name: &str, picked: &mut Option<String>, seen: &mut Vec<Role>) {
    let m = classify(name);
    let (title, description) = role_keys(m.role);
    let mut label = t.t(title);
    if let Some(index) = m.expert {
        label = format!("{label} · {}", t.ta("structure-expert-index", &[("index", index.into())]));
    }
    if m.param == Param::Bias {
        label = format!("{label} · {}", t.t("structure-bias"));
    }
    let selected = picked.as_deref() == Some(name);
    if ui.selectable_label(selected, RichText::new(label).strong()).clicked() {
        *picked = Some(name.to_owned());
    }
    if !seen.contains(&m.role) {
        ui.label(RichText::new(t.t(description)).small().weak());
        seen.push(m.role);
    }
    ui.add_space(2.0);
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
