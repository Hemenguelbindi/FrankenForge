//! The build report card: the core's block map (folded ranges, status, source, weight delta),
//! removed parts, tensors outside the blocks, config.json changes, warnings, and an explicit
//! note that model behavior was not evaluated. Everything comes from `ff_surgery::report`;
//! this file only lays it out and translates the fixed words.

use eframe::egui::{self, RichText};
use ff_surgery::report::{Delta, FileCheck, Origin, Report, Row, Status};

use super::palette::Palette;
use super::state::Forge;
use super::widgets::{card, muted, title};
use crate::i18n::I18n;

pub fn show(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &Forge) {
    let Some(r) = state.report() else { return };
    card(ui, pal, None, |ui| {
        title(ui, pal, t.t("forge-report-title"));
        let behavior = RichText::new(t.t("forge-report-behavior")).strong().color(pal.error);
        ui.label(behavior);
        let file = match r.file {
            FileCheck::Verified => t.t("forge-report-file-verified"),
            FileCheck::NotWritten => t.t("forge-report-file-planned"),
        };
        muted(ui, pal, file);
        ui.add_space(6.0);
        for (i, stack) in r.stacks.iter().enumerate() {
            let args = [
                ("stack", stack.name.clone().into()),
                ("before", stack.blocks_before.into()),
                ("after", stack.blocks_after.into()),
            ];
            ui.label(RichText::new(t.ta("forge-report-stack", &args)).strong().color(pal.text));
            egui::Grid::new(("forge-report", i)).striped(true).spacing([14.0, 4.0]).show(ui, |ui| {
                for row in stack.rows() {
                    let color = if row.status == Status::Unchanged { pal.base } else { pal.donor };
                    ui.label(RichText::new(row.range()).monospace().color(pal.text));
                    ui.label(RichText::new(&row.source).monospace().color(color));
                    ui.label(RichText::new(status_text(t, &row)).color(pal.text));
                    ui.label(row.delta.as_ref().map_or_else(String::new, |d| delta_text(t, d)));
                    ui.end_row();
                }
            });
            ui.add_space(4.0);
        }
        details(ui, t, pal, r);
    });
}

fn details(ui: &mut egui::Ui, t: &I18n, pal: Palette, r: &Report) {
    let line = |ui: &mut egui::Ui, text: String| ui.label(RichText::new(text).color(pal.text));
    if !r.removed_blocks.is_empty() {
        let names: Vec<String> =
            r.removed_blocks.iter().map(|b| format!("{}.{}", b.stack, b.index)).collect();
        line(ui, t.ta("forge-report-removed-blocks", &[("names", names.join(", ").into())]));
    }
    if !r.removed_tensors.is_empty() {
        line(ui, t.ta("forge-report-removed-tensors", &[("names", r.removed_tensors.join(", ").into())]));
    }
    for o in &r.outside {
        let name = ("name", o.name.clone().into());
        let text = match (&o.origin, &o.delta) {
            (Origin::Generated, _) => t.ta("forge-report-outside-generated", &[name]),
            (Origin::Source { alias, .. }, _) if o.unchanged => {
                t.ta("forge-report-outside-unchanged", &[name, ("source", alias.clone().into())])
            }
            (Origin::Source { alias, name: from }, Some(d)) => {
                let delta = match d {
                    Delta::NotComputed => String::new(),
                    d => delta_text(t, d),
                };
                let args = [name, ("source", format!("{alias}:{from}").into()), ("delta", delta.into())];
                t.ta("forge-report-outside-replaced", &args)
            }
            (Origin::Source { alias, name: from }, None) => {
                t.ta("forge-report-outside-added", &[name, ("source", format!("{alias}:{from}").into())])
            }
        };
        line(ui, text);
    }
    match (r.config_present, r.config_changes.is_empty()) {
        (false, _) => line(ui, t.t("forge-report-config-none")),
        (true, true) => line(ui, t.t("forge-report-config-unchanged")),
        (true, false) => {
            let shown = line(ui, t.t("forge-report-config-changed"));
            for c in &r.config_changes {
                ui.label(RichText::new(c).monospace().small());
            }
            shown
        }
    };
    if !r.warnings.is_empty() {
        muted(ui, pal, t.t("forge-report-warnings"));
        for w in &r.warnings {
            ui.label(RichText::new(w).monospace().small());
        }
    }
    if r.experimental.is_some() {
        ui.label(RichText::new(t.t("forge-report-experimental")).strong().color(pal.error));
    }
}

fn status_text(t: &I18n, row: &Row) -> String {
    match &row.status {
        Status::Unchanged => t.t("forge-report-status-unchanged"),
        Status::Shifted { .. } => t.t("forge-report-status-shifted"),
        Status::Replaced { .. } => t.ta("forge-report-status-replaced", &[("total", row.tensors.into())]),
        Status::PartiallyReplaced { replaced, total } => t.ta(
            "forge-report-status-partial",
            &[("replaced", (*replaced).into()), ("total", (*total).into())],
        ),
        Status::Added => t.t("forge-report-status-added"),
        Status::Generated => t.t("forge-report-status-generated"),
    }
}

fn delta_text(t: &I18n, d: &Delta) -> String {
    match d {
        Delta::NotComputed => t.t("forge-report-delta-not-computed"),
        Delta::Unknown { reason } => t.ta("forge-report-delta-unknown", &[("reason", reason.clone().into())]),
        Delta::Relative { value } => {
            t.ta("forge-report-delta", &[("percent", format!("{:.2}%", value * 100.0).into())])
        }
    }
}
