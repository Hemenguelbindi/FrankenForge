//! Small building blocks of the Forge page, all colored from the `Palette`.

use eframe::egui::{self, Color32, CornerRadius, Margin, Pos2, Rect, RichText, Stroke, vec2};

use super::palette::Palette;
use crate::failure::Failure;
use crate::i18n::I18n;
use crate::views;

/// A raised panel; `accent` draws a role-colored bar along its left edge.
pub fn card<R>(
    ui: &mut egui::Ui,
    pal: Palette,
    accent: Option<Color32>,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let inner = egui::Frame::new()
        .fill(pal.card)
        .stroke(Stroke::new(1.0, pal.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin { left: 16, right: 14, top: 12, bottom: 12 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        });
    if let Some(color) = accent {
        let r = inner.response.rect;
        let bar = Rect::from_min_max(
            Pos2::new(r.left() + 1.0, r.top() + 8.0),
            Pos2::new(r.left() + 4.0, r.bottom() - 8.0),
        );
        ui.painter().rect_filled(bar, 1.5, color);
    }
    inner.inner
}

/// A drawn check mark; glyphs like U+2713 may be missing from the bundled font.
pub fn check(ui: &mut egui::Ui, pal: Palette) {
    let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), egui::Sense::hover());
    let c = r.center();
    let pts = vec![c + vec2(-4.5, 0.0), c + vec2(-1.5, 3.0), c + vec2(4.5, -3.5)];
    ui.painter().add(egui::Shape::line(pts, Stroke::new(2.0, pal.ok)));
}

pub fn title(ui: &mut egui::Ui, pal: Palette, text: String) {
    ui.label(RichText::new(text).size(16.0).strong().color(pal.text));
}

pub fn muted(ui: &mut egui::Ui, pal: Palette, text: String) {
    ui.label(RichText::new(text).color(pal.muted));
}

/// Two columns when each gets at least `min_each`, otherwise one under the other in the same order.
pub fn pair<T>(
    ui: &mut egui::Ui,
    min_each: f32,
    state: &mut T,
    left: impl FnOnce(&mut egui::Ui, &mut T),
    right: impl FnOnce(&mut egui::Ui, &mut T),
) {
    if ui.available_width() >= 2.0 * min_each + 12.0 {
        ui.columns(2, |cols| {
            if let [a, b] = cols {
                left(a, state);
                right(b, state);
            }
        });
    } else {
        left(ui, state);
        ui.add_space(10.0);
        right(ui, state);
    }
}

/// A localized explanation with the core's original error text under "technical details".
/// Red is kept to the outline and the title, so an error never floods the page.
pub fn failure_box(ui: &mut egui::Ui, t: &I18n, pal: Palette, title: String, failure: &Failure, id: &str) {
    egui::Frame::new()
        .fill(pal.tint(pal.error))
        .stroke(Stroke::new(1.0, pal.error))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(vec2(16.0, 16.0), egui::Sense::hover());
                ui.painter().circle_stroke(r.center(), 7.0, Stroke::new(1.5, pal.error));
                ui.painter().text(
                    r.center(),
                    egui::Align2::CENTER_CENTER,
                    "!",
                    egui::FontId::proportional(11.0),
                    pal.error,
                );
                ui.label(RichText::new(title).strong().color(pal.error));
            });
            ui.label(RichText::new(t.t(failure.kind.message_key())).color(pal.text));
            views::details(ui, t, id, &failure.detail);
        });
}
