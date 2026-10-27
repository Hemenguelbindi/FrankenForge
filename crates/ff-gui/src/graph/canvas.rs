//! The free-form graph canvas: circles sized by parameters, coloured by part of the model, linked
//! along the data path. Wheel zooms, dragging the background pans, dragging a circle moves it,
//! clicking a circle selects it for the "what is inside" panel.

use std::path::PathBuf;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2, vec2};

use super::inside::InsideState;
use super::layout::place;
use super::net::{Landmarks, Link, Net, What, build, landmarks};
use crate::components::Kind;
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::structure::role_keys;

#[derive(Default)]
pub struct GraphUi {
    /// The model the cached graph was built for.
    key: Option<PathBuf>,
    pub net: Net,
    pub marks: Option<Landmarks>,
    pos: Vec<[f32; 2]>,
    radius: Vec<f32>,
    pan: Vec2,
    zoom: f32,
    fitted: bool,
    dragging: Option<usize>,
    pub selected: Option<usize>,
    pub inside: InsideState,
}

impl GraphUi {
    /// Rebuilds the graph only when another model is open; layout runs once per model.
    pub fn sync(&mut self, inspected: &Inspected) {
        if self.key.as_ref() == Some(&inspected.path) {
            return;
        }
        let towers = super::towers(inspected);
        let net = build(inspected, &towers);
        let max = net.nodes.iter().map(|n| n.params).max().unwrap_or(1).max(1) as f32;
        self.radius = net.nodes.iter().map(|n| 7.0 + 38.0 * (n.params as f32 / max).sqrt()).collect();
        self.pos = place(&net, &self.radius);
        self.marks = Some(landmarks(&net));
        // The language entry is where reading starts.
        self.selected = self.marks.as_ref().and_then(|m| m.entry);
        self.net = net;
        self.key = Some(inspected.path.clone());
        self.fitted = false;
        self.inside = InsideState::default();
    }
}

pub fn tower_color(kind: Kind, dark: bool) -> Color32 {
    let (d, l) = match kind {
        Kind::Language => ((96, 165, 250), (37, 99, 235)),
        Kind::Vision => ((74, 222, 128), (22, 163, 74)),
        Kind::Audio => ((251, 146, 60), (234, 88, 12)),
        Kind::Other => ((161, 161, 170), (113, 113, 122)),
    };
    let (r, g, b) = if dark { d } else { l };
    Color32::from_rgb(r, g, b)
}

/// The short name drawn on the canvas.
pub fn name(t: &I18n, what: &What) -> String {
    match what {
        What::Words(n) if n.role == crate::semantics::Role::OutputHead => t.t("net-output-words"),
        What::Words(_) => t.t("net-words"),
        What::PerLayer { .. } => t.t("net-per-layer"),
        What::Block { index, .. } => index.to_string(),
        What::Bridge(_) => t.t("net-bridge"),
        What::Output(n) | What::Input(n) => t.t(role_keys(n.role).0),
    }
}

pub fn show(ui: &mut egui::Ui, t: &I18n, g: &mut GraphUi) {
    let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let dark = ui.visuals().dark_mode;
    let text = ui.visuals().text_color();
    if !g.fitted && !g.pos.is_empty() {
        fit(g, rect);
    }
    let to_screen = |p: [f32; 2], g: &GraphUi| rect.center() + g.pan + vec2(p[0], p[1]) * g.zoom;
    let hit = |pointer: Pos2, g: &GraphUi| {
        (0..g.pos.len())
            .rev()
            .find(|&i| to_screen(g.pos[i], g).distance(pointer) <= (g.radius[i] * g.zoom).max(6.0))
    };

    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0
            && let Some(p) = resp.hover_pos()
        {
            let factor = (scroll * 0.0025).exp();
            let before = (p - rect.center() - g.pan) / g.zoom;
            g.zoom = (g.zoom * factor).clamp(0.05, 8.0);
            g.pan = p - rect.center() - before * g.zoom;
        }
    }
    let pointer = resp.interact_pointer_pos();
    if resp.drag_started() {
        g.dragging = pointer.and_then(|p| hit(p, g));
    }
    if resp.dragged() {
        let d = resp.drag_delta();
        match g.dragging {
            Some(i) => {
                g.pos[i][0] += d.x / g.zoom;
                g.pos[i][1] += d.y / g.zoom;
            }
            None => g.pan += d,
        }
    }
    if resp.drag_stopped() {
        g.dragging = None;
    }
    if resp.clicked()
        && let Some(i) = pointer.and_then(|p| hit(p, g))
    {
        g.selected = Some(i);
    }
    let hovered = resp.hover_pos().and_then(|p| hit(p, g));

    let p = ui.painter_at(rect);
    let line = ui.visuals().widgets.noninteractive.fg_stroke.color;
    for &(a, b, link) in &g.net.links {
        let near = g.selected.is_some_and(|s| s == a || s == b) || hovered.is_some_and(|h| h == a || h == b);
        let alpha = match (link, near) {
            (_, true) => 220,
            (Link::Flow, false) => 120,
            (Link::Feed, false) => 22,
        };
        let stroke = Stroke::new(if near { 2.0 } else { 1.2 }, line.gamma_multiply(alpha as f32 / 255.0));
        p.line_segment([to_screen(g.pos[a], g), to_screen(g.pos[b], g)], stroke);
    }
    for (i, n) in g.net.nodes.iter().enumerate() {
        let c = to_screen(g.pos[i], g);
        let r = (g.radius[i] * g.zoom).max(4.0);
        if !rect.expand(r).contains(c) {
            continue;
        }
        p.circle_filled(c, r, tower_color(n.tower, dark));
        if matches!(n.what, What::Block { shared_memory: true, .. }) {
            p.circle_stroke(c, r + 2.0, Stroke::new(1.5, ui.visuals().warn_fg_color));
        }
        if g.selected == Some(i) || hovered == Some(i) {
            p.circle_stroke(c, r + 4.0, Stroke::new(2.5, text));
        }
        let label = name(t, &n.what);
        if matches!(n.what, What::Block { .. }) {
            if r >= 9.0 || hovered == Some(i) {
                p.text(c, Align2::CENTER_CENTER, label, FontId::proportional(12.0), Color32::BLACK);
            }
        } else {
            p.text(c + vec2(0.0, r + 4.0), Align2::CENTER_TOP, label, FontId::proportional(15.0), text);
        }
    }
    if let Some(i) = hovered {
        resp.on_hover_text(tip(t, g, i));
    }
    legend(ui, t, rect, dark);
}

fn tip(t: &I18n, g: &GraphUi, i: usize) -> String {
    let n = &g.net.nodes[i];
    match &n.what {
        What::Block { index, kind, .. } => {
            let mut s =
                t.ta("net-block-tip", &[("index", (*index).into()), ("params", t.count(n.params).into())]);
            if let Some(k) = kind {
                s = format!("{s}\n{k}");
            }
            s
        }
        other => format!("{}\n{}", name(t, other), t.count(n.params)),
    }
}

/// Zoom and pan so the whole graph fits the canvas with a margin.
fn fit(g: &mut GraphUi, rect: Rect) {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for (p, r) in g.pos.iter().zip(&g.radius) {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k] - r);
            hi[k] = hi[k].max(p[k] + r);
        }
    }
    let size = vec2(hi[0] - lo[0], hi[1] - lo[1]) + vec2(120.0, 120.0);
    g.zoom = (rect.width() / size.x).min(rect.height() / size.y).clamp(0.05, 2.0);
    g.pan = -vec2((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0) * g.zoom;
    g.fitted = true;
}

fn legend(ui: &egui::Ui, t: &I18n, rect: Rect, dark: bool) {
    let p = ui.painter_at(rect);
    let text = ui.visuals().text_color();
    let weak = ui.visuals().weak_text_color();
    let mut at = rect.left_top() + vec2(12.0, 12.0);
    for (kind, key) in [
        (Kind::Language, "component-language"),
        (Kind::Vision, "component-vision"),
        (Kind::Audio, "component-audio"),
        (Kind::Other, "component-other"),
    ] {
        p.circle_filled(at + vec2(6.0, 8.0), 6.0, tower_color(kind, dark));
        p.text(at + vec2(18.0, 0.0), Align2::LEFT_TOP, t.t(key), FontId::proportional(14.0), text);
        at.y += 20.0;
    }
    for key in ["net-legend-size", "net-legend-shared", "net-legend-controls"] {
        p.text(at, Align2::LEFT_TOP, t.t(key), FontId::proportional(13.0), weak);
        at.y += 18.0;
    }
}
