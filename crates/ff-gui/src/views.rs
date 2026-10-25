//! Central-panel views: the component tree, the tensor table, search results, config.json, and
//! the shared "technical details" block.

use std::collections::BTreeMap;

use eframe::egui::{self, RichText};
use ff_core::{Block, TensorInfo};

use crate::components::Component;
use crate::i18n::I18n;
use crate::inspected::{Config, Inspected};
use crate::semantics::{Group, classify};

/// Larger "outside blocks" lists stay collapsed until the user opens them.
pub(crate) const AUTO_OPEN_ROWS: usize = 200;

/// Original, untranslated error text: selectable so it can be copied into a bug report.
pub(crate) fn details(ui: &mut egui::Ui, t: &I18n, id: &str, detail: &str) {
    egui::CollapsingHeader::new(t.t("error-details")).id_salt(id).show(ui, |ui| {
        ui.add(egui::Label::new(RichText::new(detail).monospace()).selectable(true));
    });
}

/// A weak label and a strong value on one wrapped row: long values wrap inside the panel.
pub(crate) fn stat(ui: &mut egui::Ui, label: String, value: String) -> egui::Response {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).weak());
        ui.label(RichText::new(value).strong());
    })
    .response
}

/// Stacks -> blocks -> groups -> tensors of one component. Collapsed bodies are not built, so a
/// checkpoint with 100k+ tensors stays responsive until a block is opened.
pub(crate) fn component_tree(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    c: &Component,
    picked: &mut Option<String>,
) {
    let tensors = &inspected.model.tensors;
    ui.heading(t.t(c.kind.message_keys().0));
    ui.label(RichText::new(t.t("tree-explainer")).weak());
    ui.add_space(6.0);
    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        for stack in &c.stacks {
            let title = t.ta(
                "stack-blocks",
                &[("name", stack.name.clone().into()), ("count", stack.blocks.len().into())],
            );
            egui::CollapsingHeader::new(title).id_salt(&stack.name).default_open(c.stacks.len() == 1).show(
                ui,
                |ui| {
                    for &i in &stack.blocks {
                        let block = &inspected.layout.blocks[i];
                        let title = t.ta(
                            "tree-block",
                            &[("index", block.index.into()), ("count", block.tensors.len().into())],
                        );
                        egui::CollapsingHeader::new(title)
                            .id_salt(&block.prefix)
                            .show(ui, |ui| block_groups(ui, t, inspected, block, picked));
                    }
                },
            );
        }
        if !c.loose.is_empty() {
            egui::CollapsingHeader::new(t.ta("tree-loose", &[("count", c.loose.len().into())]))
                .id_salt((c.kind, "loose"))
                // A plain grid lays out every row, so only small lists open by themselves.
                .default_open(c.stacks.is_empty() && c.loose.len() <= AUTO_OPEN_ROWS)
                .show(ui, |ui| {
                    let rows = c.loose.iter().map(|n| (n.as_str(), n.as_str(), &tensors[n]));
                    tensor_grid(ui, t, &format!("{:?}-loose", c.kind), rows, picked);
                });
        }
    });
}

/// A block's tensors grouped by what they do (attention, MLP, experts, norms, other). Grouping
/// uses the name rules only; the raw names stay visible in each row.
fn block_groups(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    block: &Block,
    picked: &mut Option<String>,
) {
    let mut groups: BTreeMap<Group, Vec<&String>> = BTreeMap::new();
    for name in &block.tensors {
        groups.entry(classify(name).role.group()).or_default().push(name);
    }
    for (group, names) in groups {
        let title = format!("{} ({})", t.t(group.message_key()), names.len());
        egui::CollapsingHeader::new(title).id_salt((&block.prefix, group)).default_open(true).show(
            ui,
            |ui| {
                // Names are untrusted: never slice by byte length, fall back to the full name.
                let rows = names.iter().map(|n| {
                    (
                        n.strip_prefix(block.prefix.as_str()).unwrap_or(n),
                        n.as_str(),
                        &inspected.model.tensors[*n],
                    )
                });
                tensor_grid(ui, t, &format!("{}{group:?}", block.prefix), rows, picked);
            },
        );
    }
}

/// Rows of (shown name, full name, info); the full name is always one hover away, and a click
/// opens the tensor in the inspector.
fn tensor_grid<'a>(
    ui: &mut egui::Ui,
    t: &I18n,
    id: &str,
    rows: impl Iterator<Item = (&'a str, &'a str, &'a TensorInfo)>,
    picked: &mut Option<String>,
) {
    egui::Grid::new(id).striped(true).spacing([24.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new(t.t("grid-name")).strong());
        ui.label(RichText::new(t.t("grid-type")).strong());
        ui.label(RichText::new(t.t("grid-shape")).strong());
        ui.end_row();
        for (shown, full, info) in rows {
            let selected = picked.as_deref() == Some(full);
            if ui.selectable_label(selected, RichText::new(shown).monospace()).on_hover_text(full).clicked() {
                *picked = Some(full.to_owned());
            }
            ui.monospace(format!("{:?}", info.dtype));
            ui.monospace(format!("{:?}", info.shape));
            ui.end_row();
        }
    });
}

/// Flat search result; only visible rows are laid out.
pub(crate) fn hits(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    hits: &[String],
    picked: &mut Option<String>,
) {
    ui.label(t.ta("search-matches", &[("count", hits.len().into())]));
    // Rows are selectable buttons: text plus button padding, never less than the interact height.
    // `show_rows` needs that exact height, so it is also forced as the buttons' minimum.
    let text_height = ui.text_style_height(&egui::TextStyle::Monospace);
    let row_height = (text_height + 2.0 * ui.spacing().button_padding.y).max(ui.spacing().interact_size.y);
    egui::ScrollArea::both().auto_shrink(false).show_rows(ui, row_height, hits.len(), |ui, range| {
        // Unwrapped rows keep each one a single line.
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        for name in &hits[range] {
            let info = &inspected.model.tensors[name];
            let text = RichText::new(format!("{name}  {:?} {:?}", info.dtype, info.shape)).monospace();
            let selected = picked.as_deref() == Some(name.as_str());
            let row = egui::Button::selectable(selected, text).min_size(egui::vec2(0.0, row_height));
            if ui.add(row).clicked() {
                *picked = Some(name.clone());
            }
        }
    });
}

pub(crate) fn config(ui: &mut egui::Ui, t: &I18n, config: &Config) {
    match config {
        Config::Missing => {
            ui.label(t.t("config-missing"));
        }
        Config::Invalid { problem, detail } => {
            ui.colored_label(ui.visuals().error_fg_color, t.t(problem.message_key()));
            details(ui, t, "config-error-details", detail);
        }
        Config::Json { path, text } => {
            let source = t.ta("config-source", &[("path", path.display().to_string().into())]);
            ui.label(RichText::new(source).small().weak());
            egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                // `&str` is a read-only text buffer: selectable and copyable, not editable.
                ui.add(
                    egui::TextEdit::multiline(&mut text.as_str()).code_editor().desired_width(f32::INFINITY),
                );
            });
        }
    }
}
