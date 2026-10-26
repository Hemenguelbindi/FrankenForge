//! Draws the graph: each tower as input, a strip of blocks and output, with arrows between them
//! and into the language part; a chosen block as branches off the stream (reads it, adds back).
//! Only layout and painting here; what is drawn comes from `graph`.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, vec2};

use super::{BlockGraph, Flow, Node, StackGraph, Tower};
use crate::components::Kind;
use crate::i18n::I18n;
use crate::structure::{group_keys, role_keys};

const CELL: egui::Vec2 = egui::vec2(16.0, 20.0);
const GAP: f32 = 3.0;

struct Colors {
    text: Color32,
    weak: Color32,
    line: Color32,
    fill: Color32,
    accent: Color32,
    warn: Color32,
}

fn colors(ui: &egui::Ui) -> Colors {
    let v = ui.visuals();
    Colors {
        text: v.text_color(),
        weak: v.weak_text_color(),
        line: v.widgets.noninteractive.fg_stroke.color,
        fill: v.faint_bg_color,
        accent: v.selection.bg_fill,
        warn: v.warn_fg_color,
    }
}

/// Every tower, the current one highlighted; returns a clicked block (index into the layout).
pub fn model(
    ui: &mut egui::Ui,
    t: &I18n,
    towers: &[Tower],
    current: Kind,
    chosen: Option<usize>,
) -> Option<usize> {
    let c = colors(ui);
    let language = towers.iter().find(|w| w.kind == Kind::Language).and_then(|w| w.width);
    let mut clicked = None;
    for tower in towers {
        let stroke =
            if tower.kind == current { Stroke::new(2.0, c.accent) } else { Stroke::new(1.0, c.line) };
        egui::Frame::new().stroke(stroke).corner_radius(6).inner_margin(8).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let blocks: usize = tower.stacks.iter().map(|s| s.blocks.len()).sum();
            let mut title = t.ta(
                "graph-tower",
                &[("part", t.t(tower.kind.message_keys().0).into()), ("blocks", blocks.into())],
            );
            if let Some(w) = tower.width {
                title = format!("{title} · {}", t.ta("graph-width", &[("width", w.into())]));
            }
            ui.label(RichText::new(title).strong().color(c.text));
            ui.horizontal_wrapped(|ui| {
                for n in &tower.inputs {
                    chip(ui, t, &c, n, None);
                    arrow(ui, &c);
                }
                if tower.inputs.is_empty() {
                    ui.label(RichText::new(t.t("graph-input-unknown")).small().color(c.weak));
                    arrow(ui, &c);
                }
                for s in &tower.stacks {
                    if let Some(i) = strip(ui, t, &c, s, chosen) {
                        clicked = Some(i);
                    }
                    arrow(ui, &c);
                }
                for n in &tower.outputs {
                    chip(ui, t, &c, n, None);
                    arrow(ui, &c);
                }
                for n in &tower.bridges {
                    chip(ui, t, &c, n, language);
                    arrow(ui, &c);
                }
                let end = match tower.kind {
                    Kind::Language => "graph-output",
                    _ if tower.bridges.is_empty() => "graph-joins-unknown",
                    _ => "graph-into-language",
                };
                ui.label(RichText::new(t.t(end)).small().color(c.weak));
            });
            notes(ui, t, &c, tower);
        });
        ui.add_space(6.0);
    }
    clicked
}

fn notes(ui: &mut egui::Ui, t: &I18n, c: &Colors, tower: &Tower) {
    let note = |ui: &mut egui::Ui, text: String| ui.label(RichText::new(text).small().color(c.weak));
    for n in &tower.per_layer {
        note(
            ui,
            t.ta("graph-per-layer", &[("name", n.name.clone().into()), ("shape", shape(&n.shape).into())]),
        );
    }
    for s in &tower.stacks {
        if let Some(k) = s.kv_shared {
            let first = s.blocks.len() - k;
            let args = [("first", first.into()), ("last", (s.blocks.len() - 1).into()), ("count", k.into())];
            ui.label(RichText::new(t.ta("graph-kv-shared", &args)).small().color(c.warn));
        }
        let kinds = layer_kinds(s);
        if kinds.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                note(ui, t.t("graph-layer-types"));
                for (i, kind) in kinds.iter().enumerate() {
                    let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
                    ui.painter().rect_filled(r, 2.0, kind_fill(c, i));
                    note(ui, kind.clone());
                }
            });
        }
    }
    if tower.other > 0 {
        note(ui, t.ta("graph-other-loose", &[("count", tower.other.into())]));
    }
}

/// Distinct layer types in order of first appearance.
fn layer_kinds(s: &StackGraph) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in &s.layer_types {
        if !out.contains(k) {
            out.push(k.clone());
        }
    }
    out
}

fn kind_fill(c: &Colors, i: usize) -> Color32 {
    match i {
        0 => c.fill,
        1 => c.accent,
        _ => c.warn,
    }
}

/// A small framed box: role title and stored shape (or the width it maps to the language part).
fn chip(ui: &mut egui::Ui, t: &I18n, c: &Colors, n: &Node, language: Option<usize>) {
    let detail = match (&n.shape[..], language) {
        ([a, b], Some(w)) if *a == w => t.ta("graph-flow", &[("from", (*b).into()), ("to", w.into())]),
        ([a, b], Some(w)) if *b == w => t.ta("graph-flow", &[("from", (*a).into()), ("to", w.into())]),
        _ => shape(&n.shape),
    };
    egui::Frame::new().stroke(Stroke::new(1.0, c.line)).corner_radius(4).inner_margin(4).show(ui, |ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new(t.t(role_keys(n.role).0)).small().color(c.text));
            ui.label(RichText::new(detail).small().monospace().color(c.weak));
        })
        .response
        .on_hover_text(&n.name);
    });
}

fn shape(s: &[usize]) -> String {
    s.iter().map(usize::to_string).collect::<Vec<_>>().join(" × ")
}

fn arrow(ui: &mut egui::Ui, c: &Colors) {
    let (r, _) = ui.allocate_exact_size(vec2(22.0, 14.0), Sense::hover());
    let (a, b) = (r.left_center(), r.right_center() - vec2(1.0, 0.0));
    let p = ui.painter();
    p.line_segment([a, b], Stroke::new(1.5, c.line));
    p.line_segment([b, b + vec2(-5.0, -4.0)], Stroke::new(1.5, c.line));
    p.line_segment([b, b + vec2(-5.0, 4.0)], Stroke::new(1.5, c.line));
}

/// One cell per block, wrapped to the width left on the line; kv-shared blocks get a warning
/// border, the chosen one the accent. Returns a clicked block.
fn strip(ui: &mut egui::Ui, t: &I18n, c: &Colors, s: &StackGraph, chosen: Option<usize>) -> Option<usize> {
    let n = s.blocks.len();
    let per_row =
        (((ui.available_width() - 30.0).max(CELL.x) + GAP) / (CELL.x + GAP)).floor().max(1.0) as usize;
    let cols = n.min(per_row);
    let rows = n.div_ceil(cols);
    let size = vec2(cols as f32 * (CELL.x + GAP), rows as f32 * (CELL.y + GAP));
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    let kinds = layer_kinds(s);
    let shared_from = s.kv_shared.map(|k| n - k);
    let mut clicked = None;
    for (pos, &layout_index) in s.blocks.iter().enumerate() {
        let at = area.min + vec2((pos % cols) as f32 * (CELL.x + GAP), (pos / cols) as f32 * (CELL.y + GAP));
        let r = Rect::from_min_size(at, CELL);
        let kind = s.layer_types.get(pos);
        let fill = kind.and_then(|k| kinds.iter().position(|x| x == k)).map_or(c.fill, |i| kind_fill(c, i));
        let resp = ui.interact(r, ui.make_persistent_id(("graph-cell", &s.name, pos)), Sense::click());
        let p = ui.painter();
        p.rect_filled(r, 2.0, fill);
        let border = if Some(layout_index) == chosen {
            Stroke::new(2.0, c.text)
        } else if shared_from.is_some_and(|f| pos >= f) {
            Stroke::new(1.5, c.warn)
        } else {
            Stroke::new(1.0, c.line)
        };
        p.rect_stroke(r, 2.0, border, StrokeKind::Inside);
        p.text(r.center(), Align2::CENTER_CENTER, pos.to_string(), FontId::monospace(8.0), c.text);
        let mut tip = t.ta("graph-block-tip", &[("index", pos.into())]);
        if let Some(k) = kind {
            tip = format!("{tip}\n{k}");
        }
        if resp.on_hover_text(tip).clicked() {
            clicked = Some(layout_index);
        }
    }
    clicked
}

const BOX_W: f32 = 250.0;
const LINE: f32 = 16.0;

/// The chosen block: the stream on top, each branch below it reading from the stream and adding
/// its result back. Returns a clicked tensor name.
pub fn block(ui: &mut egui::Ui, t: &I18n, g: &BlockGraph) -> Option<String> {
    let c = colors(ui);
    if g.branches.is_empty() {
        ui.label(RichText::new(t.t("graph-block-empty")).color(c.weak));
        return None;
    }
    let rows = |b: &super::Branch| b.matrices.len() + usize::from(b.small > 0);
    let tallest = g.branches.iter().map(rows).max().unwrap_or(0) as f32;
    let width = (g.branches.len() as f32 * (BOX_W + 24.0) + 40.0).max(ui.available_width());
    let height = 64.0 + 28.0 + tallest * LINE + 24.0;
    let mut clicked = None;
    egui::ScrollArea::horizontal().id_salt("graph-block").show(ui, |ui| {
        let (area, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
        let p = ui.painter().clone();
        let stream_y = area.top() + 24.0;
        let stream = Stroke::new(3.0, c.accent);
        let end = Pos2::new(area.right() - 8.0, stream_y);
        p.line_segment([Pos2::new(area.left() + 4.0, stream_y), end], stream);
        p.line_segment([end, end + vec2(-8.0, -6.0)], stream);
        p.line_segment([end, end + vec2(-8.0, 6.0)], stream);
        let label = match g.width {
            Some(w) => t.ta("graph-stream", &[("width", w.into())]),
            None => t.t("graph-stream-unknown"),
        };
        p.text(
            Pos2::new(area.left() + 4.0, stream_y - 6.0),
            Align2::LEFT_BOTTOM,
            label,
            FontId::proportional(12.0),
            c.text,
        );
        if g.norms > 0 {
            let text = t.ta("graph-norms", &[("count", g.norms.into())]);
            p.text(
                Pos2::new(area.right() - 8.0, stream_y - 6.0),
                Align2::RIGHT_BOTTOM,
                text,
                FontId::proportional(11.0),
                c.weak,
            );
        }
        for (i, b) in g.branches.iter().enumerate() {
            let left = area.left() + 24.0 + i as f32 * (BOX_W + 24.0);
            let top = stream_y + 40.0;
            let r = Rect::from_min_size(Pos2::new(left, top), vec2(BOX_W, 28.0 + rows(b) as f32 * LINE));
            p.rect_filled(r, 6.0, c.fill);
            p.rect_stroke(r, 6.0, Stroke::new(1.0, c.line), StrokeKind::Inside);
            // Reads the stream on the left, adds its result back on the right.
            let (down, up) = (left + 20.0, left + BOX_W - 20.0);
            p.line_segment([Pos2::new(down, stream_y), Pos2::new(down, top)], Stroke::new(1.5, c.line));
            p.line_segment([Pos2::new(up, top), Pos2::new(up, stream_y + 7.0)], Stroke::new(1.5, c.line));
            p.circle_stroke(Pos2::new(up, stream_y), 7.0, Stroke::new(1.5, c.text));
            p.text(Pos2::new(up, stream_y), Align2::CENTER_CENTER, "+", FontId::proportional(13.0), c.text);
            p.text(
                r.left_top() + vec2(8.0, 6.0),
                Align2::LEFT_TOP,
                t.t(group_keys(b.group).0),
                FontId::proportional(13.0),
                c.text,
            );
            for (row, m) in b.matrices.iter().enumerate() {
                let line = Rect::from_min_size(
                    r.left_top() + vec2(4.0, 26.0 + row as f32 * LINE),
                    vec2(BOX_W - 8.0, LINE),
                );
                let resp =
                    ui.interact(line, ui.make_persistent_id(("graph-matrix", &m.name)), Sense::click());
                if resp.hovered() {
                    p.rect_filled(line, 3.0, c.accent.gamma_multiply(0.4));
                }
                let mut text = format!("{}  {}", t.t(role_keys(m.role).0), flow_text(t, &m.flow));
                if m.count > 1 {
                    text = format!("{text}  × {}", m.count);
                }
                p.text(
                    line.left_center() + vec2(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    text,
                    FontId::proportional(11.5),
                    c.text,
                );
                if resp.on_hover_text(&m.name).clicked() {
                    clicked = Some(m.name.clone());
                }
            }
            if b.small > 0 {
                let at = r.left_top() + vec2(8.0, 26.0 + b.matrices.len() as f32 * LINE + LINE / 2.0);
                let text = t.ta("graph-small", &[("count", b.small.into())]);
                p.text(at, Align2::LEFT_CENTER, text, FontId::proportional(11.0), c.weak);
            }
        }
    });
    clicked
}

fn flow_text(t: &I18n, f: &Flow) -> String {
    match f {
        Flow::Reads { from, to } | Flow::Writes { from, to } => {
            t.ta("graph-flow", &[("from", (*from).into()), ("to", (*to).into())])
        }
        Flow::Shape(s) => shape(s),
    }
}
