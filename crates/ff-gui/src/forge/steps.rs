//! The step strip Base → Donor → Blocks → Build. Each step's mark comes from the real state; it is
//! drawn as a shape and named in words, so it never depends on color alone.

use eframe::egui::{self, Align2, FontId, RichText, Sense, Stroke};

use super::build::BuildError;
use super::palette::Palette;
use super::readiness::Problem;
use super::state::{BuildStatus, Forge};
use crate::i18n::I18n;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Done,
    Current,
    Waiting,
    Attention,
}

impl Mark {
    pub const KEYS: [&str; 4] =
        ["forge-step-done", "forge-step-current", "forge-step-waiting", "forge-step-attention"];

    pub fn message_key(self) -> &'static str {
        Self::KEYS[self as usize]
    }
}

pub const TITLES: [&str; 4] =
    ["forge-step-base", "forge-step-donor", "forge-step-blocks", "forge-step-build"];

pub fn marks(f: &Forge, problems: &[Problem]) -> [Mark; 4] {
    let has = |p: &dyn Fn(&Problem) -> bool| problems.iter().any(p);
    let base = if f.base().is_some() { Mark::Done } else { Mark::Current };
    let donor = match (f.base(), f.donor()) {
        (_, Some(_)) if has(&|p| *p == Problem::SameModel) => Mark::Attention,
        (_, Some(_)) => Mark::Done,
        (Some(_), None) => Mark::Current,
        (None, None) => Mark::Waiting,
    };
    let blocks =
        if has(&|p| matches!(p, Problem::AnalysisFailed | Problem::Tokenizer | Problem::PreviewFailed)) {
            Mark::Attention
        } else if matches!(f.preview(), Some(Ok(_))) {
            Mark::Done
        } else if matches!(f.analysis(), Some(Ok(_))) {
            Mark::Current
        } else {
            Mark::Waiting
        };
    let build = match f.build_status() {
        BuildStatus::Done(Ok(_)) => Mark::Done,
        BuildStatus::Done(Err(BuildError::Refused(_) | BuildError::Failed { .. })) => Mark::Attention,
        BuildStatus::Running(_) => Mark::Current,
        BuildStatus::Idle if blocks != Mark::Done => Mark::Waiting,
        BuildStatus::Idle if has(&|p| matches!(p, Problem::Output(_))) => Mark::Attention,
        BuildStatus::Idle if problems.is_empty() => Mark::Current,
        BuildStatus::Idle => Mark::Waiting,
    };
    [base, donor, blocks, build]
}

pub fn strip(ui: &mut egui::Ui, t: &I18n, pal: Palette, marks: [Mark; 4]) {
    ui.horizontal_wrapped(|ui| {
        for (i, (mark, title)) in marks.into_iter().zip(TITLES).enumerate() {
            if i > 0 {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 22.0), Sense::hover());
                let y = rect.center().y;
                ui.painter().line_segment(
                    [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                    (1.0, pal.border),
                );
            }
            // One group per step so a wrapped line never separates a mark from its words.
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.horizontal(|ui| {
                    marker(ui, pal, mark, i + 1);
                    ui.label(RichText::new(t.t(title)).strong().color(pal.text));
                    let color = if mark == Mark::Attention { pal.error } else { pal.muted };
                    ui.label(RichText::new(t.t(mark.message_key())).small().color(color));
                });
            });
        }
    });
}

/// Done: filled disc with a check. Current: thick ring with the number. Waiting: thin ring with
/// the number. Attention: ring with "!".
fn marker(ui: &mut egui::Ui, pal: Palette, mark: Mark, n: usize) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), Sense::hover());
    let (c, r, p) = (rect.center(), 10.0, ui.painter());
    let font = FontId::proportional(12.0);
    match mark {
        Mark::Done => {
            p.circle_filled(c, r, pal.ok);
            let pts = [c + egui::vec2(-4.5, 0.0), c + egui::vec2(-1.0, 3.5), c + egui::vec2(5.0, -3.5)];
            p.add(egui::Shape::line(pts.to_vec(), Stroke::new(2.0, pal.card)));
        }
        Mark::Current => {
            p.circle_stroke(c, r, Stroke::new(2.0, pal.seam));
            p.text(c, Align2::CENTER_CENTER, n.to_string(), font, pal.text);
        }
        Mark::Waiting => {
            p.circle_stroke(c, r, Stroke::new(1.0, pal.muted));
            p.text(c, Align2::CENTER_CENTER, n.to_string(), font, pal.muted);
        }
        Mark::Attention => {
            p.circle_stroke(c, r, Stroke::new(2.0, pal.error));
            p.text(c, Align2::CENTER_CENTER, "!", font, pal.error);
        }
    }
}

#[cfg(test)]
#[path = "tests/steps.rs"]
mod tests;
