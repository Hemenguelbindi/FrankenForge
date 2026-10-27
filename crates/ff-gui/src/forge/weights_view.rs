//! "Where the models differ": the core's per-block weight difference of Donor against Base,
//! one bar per block, with the part that moved most. It answers "which block is worth taking"
//! with a measurement; what a transplant does to behavior is said to be unknown. Clicking a
//! block selects it like the schema does.

use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui::{self, RichText, Sense, vec2};
use ff_surgery::compare::weights::{BlockWeights, WeightDiff, short};
use ff_surgery::report::Delta;

use super::analysis::BlockId;
use super::jobs::Meter;
use super::palette::Palette;
use super::state::Forge;
use super::widgets::{card, failure_box, muted, title};
use crate::failure::FailureKind;
use crate::i18n::I18n;

pub enum Action {
    None,
    Start,
    Stop,
    Toggle(BlockId),
}

pub fn show(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &Forge, running: Option<&Meter>) -> Action {
    card(ui, pal, None, |ui| {
        title(ui, pal, t.t("forge-diff-title"));
        muted(ui, pal, t.t("forge-diff-intro"));
        ui.add_space(6.0);
        if !matches!(state.analysis(), Some(Ok(_))) {
            muted(ui, pal, t.t("forge-diff-need-analysis"));
            return Action::None;
        }
        if let Some(m) = running {
            let total = m.total.load(Ordering::Relaxed).max(1);
            let fraction = m.done.load(Ordering::Relaxed) as f32 / total as f32;
            let percent = format!("{:.0}", fraction * 100.0);
            ui.add(
                egui::ProgressBar::new(fraction)
                    .text(t.ta("forge-diff-running", &[("percent", percent.into())])),
            );
            // The worker wakes the UI only when done; keep the bar moving meanwhile.
            ui.ctx().request_repaint_after(Duration::from_millis(100));
            return if ui.button(t.t("forge-diff-stop")).clicked() { Action::Stop } else { Action::None };
        }
        match state.weights() {
            None => start(ui, t, pal),
            Some(Err(f)) if f.kind == FailureKind::Cancelled => {
                muted(ui, pal, t.t("forge-diff-stopped"));
                start(ui, t, pal)
            }
            Some(Err(f)) => {
                failure_box(ui, t, pal, t.t("forge-diff-failed"), f, "forge-diff");
                start(ui, t, pal)
            }
            Some(Ok(d)) => map(ui, t, pal, state, d),
        }
    })
}

fn start(ui: &mut egui::Ui, t: &I18n, pal: Palette) -> Action {
    muted(ui, pal, t.t("forge-diff-start-hint"));
    if ui.button(RichText::new(t.t("forge-diff-start")).strong()).clicked() {
        Action::Start
    } else {
        Action::None
    }
}

fn relative(b: &BlockWeights) -> Option<f64> {
    match b.delta {
        Delta::Relative { value } => Some(value),
        _ => None,
    }
}

fn map(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &Forge, d: &WeightDiff) -> Action {
    if d.blocks.is_empty() {
        ui.label(RichText::new(t.t("forge-diff-none")).color(pal.text));
        return Action::None;
    }
    let measured = || d.blocks.iter().filter_map(|b| Some((b, relative(b)?)));
    let max = measured().map(|(_, v)| v).fold(0.0, f64::max);
    if let (Some((lo, lo_v)), Some((hi, hi_v))) =
        (measured().min_by(|a, b| a.1.total_cmp(&b.1)), measured().max_by(|a, b| a.1.total_cmp(&b.1)))
    {
        let args = [
            ("min", pct(lo_v).into()),
            ("min_block", lo.index.into()),
            ("max", pct(hi_v).into()),
            ("max_block", hi.index.into()),
        ];
        ui.label(RichText::new(t.ta("forge-diff-summary", &args)).strong().color(pal.text));
    }
    muted(ui, pal, t.t("forge-diff-click"));
    let mut action = Action::None;
    egui::ScrollArea::vertical().id_salt("forge-diff").max_height(360.0).show(ui, |ui| {
        egui::Grid::new("forge-diff-grid").spacing([10.0, 3.0]).show(ui, |ui| {
            for b in &d.blocks {
                let id = BlockId { stack: b.stack.clone(), index: b.index };
                let label = t.ta("forge-diff-block", &[("index", b.index.into())]);
                if ui.selectable_label(state.is_selected(&id), label).clicked() {
                    action = Action::Toggle(id);
                }
                let (r, _) = ui.allocate_exact_size(vec2(160.0, 10.0), Sense::hover());
                ui.painter().rect_filled(r, 3.0, pal.well);
                if let Some(v) = relative(b).filter(|_| max > 0.0) {
                    let w = (r.width() * (v / max) as f32).max(2.0);
                    let bar = egui::Rect::from_min_size(r.min, vec2(w, r.height()));
                    ui.painter().rect_filled(bar, 3.0, pal.donor);
                }
                let value = relative(b).map_or_else(|| t.t("forge-diff-unknown"), pct);
                ui.label(RichText::new(value).monospace().color(pal.text));
                let most = b.most_changed().map_or_else(String::new, |(n, v)| {
                    t.ta(
                        "forge-diff-most",
                        &[("part", short(n, &b.stack).to_owned().into()), ("value", pct(v).into())],
                    )
                });
                ui.label(RichText::new(most).small().color(pal.muted));
                ui.end_row();
            }
        });
    });
    if d.not_compared > 0 {
        muted(ui, pal, t.ta("forge-diff-not-compared", &[("count", d.not_compared.into())]));
    }
    ui.add_space(4.0);
    muted(ui, pal, t.t("forge-diff-how-to-read"));
    action
}

fn pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}
