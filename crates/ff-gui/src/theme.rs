//! Dark default theme with higher contrast and roomier spacing, plus a matching light theme.

use eframe::egui::{self, Color32, CornerRadius, Stroke, Style, Theme, Visuals};

const RADIUS: CornerRadius = CornerRadius::same(6);

pub fn install(ctx: &egui::Context) {
    ctx.style_mut_of(Theme::Dark, |style| {
        style.visuals = dark();
        spacing(style);
    });
    ctx.style_mut_of(Theme::Light, |style| {
        round(&mut style.visuals);
        spacing(style);
    });
    ctx.set_theme(Theme::Dark);
}

fn spacing(style: &mut Style) {
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
}

fn round(v: &mut Visuals) {
    v.window_corner_radius = RADIUS;
    v.menu_corner_radius = RADIUS;
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = RADIUS;
    }
}

fn dark() -> Visuals {
    let base = Color32::from_rgb(22, 24, 29);
    let panel = Color32::from_rgb(30, 33, 39);
    let raised = Color32::from_rgb(42, 46, 54);
    let hover = Color32::from_rgb(54, 59, 69);
    let pressed = Color32::from_rgb(66, 72, 84);
    let border = Color32::from_rgb(58, 63, 74);
    let text = Color32::from_rgb(222, 225, 230);
    let accent = Color32::from_rgb(96, 165, 250);

    let mut v = Visuals::dark();
    v.panel_fill = panel;
    v.window_fill = panel;
    v.extreme_bg_color = base;
    v.code_bg_color = base;
    v.faint_bg_color = Color32::from_rgb(36, 39, 46);
    v.hyperlink_color = accent;
    v.selection.bg_fill = Color32::from_rgb(37, 78, 140);
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = panel;
    w.noninteractive.bg_stroke = Stroke::new(1.0, border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, text);
    w.inactive.bg_fill = raised;
    w.inactive.weak_bg_fill = raised;
    w.inactive.fg_stroke = Stroke::new(1.0, text);
    w.hovered.bg_fill = hover;
    w.hovered.weak_bg_fill = hover;
    w.hovered.bg_stroke = Stroke::new(1.0, accent);
    w.hovered.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    w.active.bg_fill = pressed;
    w.active.weak_bg_fill = pressed;
    w.active.bg_stroke = Stroke::new(1.0, accent);
    w.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
    round(&mut v);
    v
}
