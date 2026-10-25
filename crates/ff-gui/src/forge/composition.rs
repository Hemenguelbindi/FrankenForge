//! The composition card: the real plan's answer to "which blocks come from the Donor and how many
//! tensors from each model".

use eframe::egui::{self, RichText, Sense, vec2};

use super::palette::Palette;
use super::state::Forge;
use super::widgets::{card, failure_box, muted, title};
use crate::i18n::I18n;
use crate::views;

pub fn show(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &mut Forge) {
    card(ui, pal, None, |ui| {
        ui.horizontal_wrapped(|ui| {
            title(ui, pal, t.t("forge-preview-title"));
            if state.selection().is_some() && ui.button(t.t("forge-select-none")).clicked() {
                state.clear_selection();
            }
        });
        let Some(sel) = state.selection() else {
            muted(ui, pal, t.t("forge-preview-empty"));
            return;
        };
        let blocks: Vec<String> = sel.blocks.iter().map(|b| format!("{} {}", b.stack, b.index)).collect();
        views::stat(ui, t.t("forge-preview-blocks"), blocks.join(", "));
        match state.preview() {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(t.t("forge-preview-running")).color(pal.text));
                });
            }
            Some(Err(failure)) => {
                failure_box(ui, t, pal, t.t("forge-preview-failed"), failure, "forge-preview")
            }
            Some(Ok(p)) => {
                share_bar(ui, pal, p.from_base, p.from_donor);
                for (key, n, color) in [
                    ("forge-preview-from-base", p.from_base, pal.base),
                    ("forge-preview-from-donor", p.from_donor, pal.donor),
                ] {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(t.t(key)).color(pal.muted));
                        ui.label(RichText::new(n.to_string()).strong().color(color));
                    });
                }
                if !p.notes.is_empty() {
                    muted(ui, pal, t.t("forge-preview-notes"));
                    for n in &p.notes {
                        ui.label(RichText::new(n).monospace().small());
                    }
                }
            }
        }
    });
}

/// Tensor share of Base and Donor; the numbers next to it carry the same information in text.
fn share_bar(ui: &mut egui::Ui, pal: Palette, base: usize, donor: usize) {
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 10.0), Sense::hover());
    let total = (base + donor).max(1) as f32;
    // A single Donor block is a tiny share; keep it visible.
    let donor_w = if donor > 0 { (r.width() * donor as f32 / total).max(6.0) } else { 0.0 };
    let split = r.right() - donor_w;
    ui.painter().rect_filled(egui::Rect::from_min_max(r.min, egui::pos2(split, r.bottom())), 3.0, pal.base);
    ui.painter().rect_filled(egui::Rect::from_min_max(egui::pos2(split, r.top()), r.max), 3.0, pal.donor);
}
