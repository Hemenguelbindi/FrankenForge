//! The guided tour for a reader who does not know ML. It uses only the checkpoint's structure:
//! no weights are read, the model is not run, and no internal signals are invented.

use eframe::egui::{self, Align2, Color32, FontId, RichText, Sense, Stroke};

use crate::components::Kind;
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::views;

/// The expandable explanations of the tour. Keeping the state here makes the accordion testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explanation {
    What,
    Blocks,
    Attention,
    Mlp,
    Normalization,
    Tensor,
}

impl Explanation {
    pub const ALL: [Explanation; 6] = [
        Explanation::What,
        Explanation::Blocks,
        Explanation::Attention,
        Explanation::Mlp,
        Explanation::Normalization,
        Explanation::Tensor,
    ];

    pub fn title_key(self) -> &'static str {
        match self {
            Explanation::What => "explain-what-title",
            Explanation::Blocks => "explain-blocks-title",
            Explanation::Attention => "explain-attention-title",
            Explanation::Mlp => "explain-mlp-title",
            Explanation::Normalization => "explain-norm-title",
            Explanation::Tensor => "explain-tensor-title",
        }
    }

    pub fn body_key(self) -> &'static str {
        match self {
            Explanation::What => "explain-what-body",
            Explanation::Blocks => "explain-blocks-body",
            Explanation::Attention => "explain-attention-body",
            Explanation::Mlp => "explain-mlp-body",
            Explanation::Normalization => "explain-norm-body",
            Explanation::Tensor => "explain-tensor-body",
        }
    }
}

/// One explanation is open at a time; choosing another closes the previous one.
#[derive(Debug, Default)]
pub struct LearnState {
    open: Option<Explanation>,
}

impl LearnState {
    #[cfg(test)]
    pub fn open(&self) -> Option<Explanation> {
        self.open
    }

    pub fn is_open(&self, explanation: Explanation) -> bool {
        self.open == Some(explanation)
    }

    pub fn toggle(&mut self, explanation: Explanation) {
        self.open = if self.open == Some(explanation) { None } else { Some(explanation) };
    }
}

/// The five steps of the route. The UI numbers them and draws the arrows.
pub(crate) const STEPS: [(&str, &str); 5] = [
    ("learn-step-text-title", "learn-step-text-body"),
    ("learn-step-pieces-title", "learn-step-pieces-body"),
    ("learn-step-numbers-title", "learn-step-numbers-body"),
    ("learn-step-blocks-title", "learn-step-blocks-body"),
    ("learn-step-next-title", "learn-step-next-body"),
];

pub fn view(ui: &mut egui::Ui, t: &I18n, inspected: &Inspected, state: &mut LearnState) {
    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        ui.set_max_width(820.0);
        ui.add_space(8.0);
        ui.heading(t.t("learn-title"));
        ui.label(t.t("learn-intro"));
        ui.add_space(4.0);
        ui.label(RichText::new(t.t("learn-scope")).small().weak());
        ui.add_space(10.0);
        facts(ui, t, inspected);
        ui.add_space(16.0);
        section(ui, t.t("learn-route-title"));
        route(ui, t);
        ui.add_space(18.0);
        section(ui, t.t("learn-explain-title"));
        for explanation in Explanation::ALL {
            accordion(ui, t, state, explanation);
            ui.add_space(4.0);
        }
        ui.add_space(12.0);
    });
}

fn section(ui: &mut egui::Ui, title: String) {
    ui.label(RichText::new(title).strong().size(16.0));
    ui.add_space(4.0);
}

/// Model facts come from the opened file only. Missing block evidence is said plainly instead of
/// pretending the tour saw a transformer.
fn facts(ui: &mut egui::Ui, t: &I18n, inspected: &Inspected) {
    egui::Frame::new().fill(ui.visuals().faint_bg_color).corner_radius(8).inner_margin(10).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(&inspected.name).strong().size(16.0));
        ui.label(RichText::new(inspected.path.display().to_string()).small().weak());
        ui.add_space(6.0);
        views::stat(ui, t.t("overview-parameters"), t.count(inspected.params));
        let named: Vec<String> = inspected
            .components
            .iter()
            .filter(|c| c.kind != Kind::Other)
            .map(|c| t.t(c.kind.message_keys().0))
            .collect();
        let parts = if named.is_empty() {
            t.t("learn-facts-no-parts")
        } else {
            t.ta("learn-facts-parts", &[("parts", named.join(", ").into())])
        };
        views::stat(ui, t.t("learn-facts-parts-label"), parts);
        let blocks: usize = inspected.components.iter().map(|c| c.block_count()).sum();
        let blocks = if blocks == 0 {
            t.t("learn-facts-no-blocks")
        } else {
            t.ta("learn-facts-blocks", &[("count", blocks.into())])
        };
        views::stat(ui, t.t("learn-facts-blocks-label"), blocks);
    });
    ui.add_space(6.0);
    ui.label(RichText::new(t.t("learn-facts-limits")).small().weak());
}

fn route(ui: &mut egui::Ui, t: &I18n) {
    for (i, (title, body)) in STEPS.iter().enumerate() {
        egui::Frame::new().fill(ui.visuals().faint_bg_color).corner_radius(6).inner_margin(8).show(
            ui,
            |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    number_badge(ui, i + 1);
                    ui.label(RichText::new(t.t(title)).strong());
                });
                ui.label(t.t(body));
            },
        );
        if i + 1 < STEPS.len() {
            arrow_down(ui);
        }
    }
}

fn number_badge(ui: &mut egui::Ui, n: usize) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), Sense::hover());
    let painter = ui.painter();
    painter.circle_filled(rect.center(), 11.0, ui.visuals().hyperlink_color);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        n.to_string(),
        FontId::proportional(13.0),
        Color32::WHITE,
    );
}

fn arrow_down(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(26.0, 16.0), Sense::hover());
    let painter = ui.painter();
    let color = ui.visuals().weak_text_color();
    let x = rect.center().x;
    painter.line_segment(
        [egui::pos2(x, rect.top() + 1.0), egui::pos2(x, rect.bottom() - 5.0)],
        Stroke::new(1.0, color),
    );
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(x - 4.0, rect.bottom() - 6.0),
            egui::pos2(x + 4.0, rect.bottom() - 6.0),
            egui::pos2(x, rect.bottom()),
        ],
        color,
        Stroke::NONE,
    ));
}

fn accordion(ui: &mut egui::Ui, t: &I18n, state: &mut LearnState, explanation: Explanation) {
    let open = state.is_open(explanation);
    let mut clicked = false;
    egui::Frame::new().fill(ui.visuals().faint_bg_color).corner_radius(6).inner_margin(8).show(ui, |ui| {
        ui.style_mut().interaction.selectable_labels = false;
        ui.set_width(ui.available_width());
        let header = ui
            .horizontal(|ui| {
                ui.label(RichText::new(t.t(explanation.title_key())).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(if open { "-" } else { "+" });
                });
            })
            .response
            .interact(Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        clicked = header.clicked();
        if open {
            ui.add_space(4.0);
            ui.label(t.t(explanation.body_key()));
            match explanation {
                Explanation::Blocks => blocks_diagram(ui, t),
                Explanation::Attention => attention_diagram(ui, t),
                Explanation::Tensor => tensor_diagram(ui, t),
                Explanation::What | Explanation::Mlp | Explanation::Normalization => {}
            }
        }
    });
    if clicked {
        state.toggle(explanation);
    }
}

/// A stack of identical-looking blocks, with arrows between them. It illustrates the file layout,
/// not activations.
fn blocks_diagram(ui: &mut egui::Ui, t: &I18n) {
    let width = ui.available_width().clamp(240.0, 380.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 128.0), Sense::hover());
    let painter = ui.painter();
    let (fill, border, text) = colors(ui);
    let box_width = (rect.width() * 0.55).clamp(120.0, 220.0);
    let left = rect.center().x - box_width / 2.0;
    let labels = [
        t.ta("learn-diagram-block", &[("index", 1.into())]),
        t.ta("learn-diagram-block", &[("index", 2.into())]),
        t.t("learn-diagram-more"),
    ];
    let mut tops = [0.0f32; 3];
    for (i, label) in labels.iter().enumerate() {
        tops[i] = rect.top() + 6.0 + i as f32 * 40.0;
        let body = egui::Rect::from_min_size(egui::pos2(left, tops[i]), egui::vec2(box_width, 26.0));
        painter.rect_filled(body, 6.0, fill);
        painter.rect_stroke(body, 6.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
        painter.text(body.center(), Align2::CENTER_CENTER, label, FontId::proportional(13.0), text);
        if i > 0 {
            arrow_vertical(painter, rect.center().x, tops[i] - 14.0, tops[i] - 2.0, border);
        }
    }
    ui.add_space(4.0);
    ui.label(RichText::new(t.t("learn-diagram-blocks-note")).small().weak());
}

/// A made-up "Anna ... she" example. The line is explicitly a principle illustration, never a
/// measured attention weight.
fn attention_diagram(ui: &mut egui::Ui, t: &I18n) {
    let width = ui.available_width().clamp(260.0, 360.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 104.0), Sense::hover());
    let painter = ui.painter();
    let (fill, border, text) = colors(ui);
    let left =
        egui::Rect::from_min_size(egui::pos2(rect.left() + 12.0, rect.top() + 34.0), egui::vec2(88.0, 28.0));
    let right = egui::Rect::from_min_size(
        egui::pos2(rect.right() - 100.0, rect.top() + 34.0),
        egui::vec2(88.0, 28.0),
    );
    for (box_rect, label) in
        [(left, t.t("learn-diagram-attn-name")), (right, t.t("learn-diagram-attn-pronoun"))]
    {
        painter.rect_filled(box_rect, 14.0, fill);
        painter.rect_stroke(box_rect, 14.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
        painter.text(box_rect.center(), Align2::CENTER_CENTER, label, FontId::proportional(14.0), text);
    }
    let y = left.center().y;
    painter.text(
        egui::pos2(rect.center().x, left.top() - 8.0),
        Align2::CENTER_CENTER,
        t.t("learn-diagram-attn-arrow"),
        FontId::proportional(12.0),
        text,
    );
    let (from, to) = (right.left() - 4.0, left.right() + 6.0);
    let mut x = from;
    while x > to {
        let end = (x - 7.0).max(to);
        painter.line_segment([egui::pos2(x, y), egui::pos2(end, y)], Stroke::new(1.0, border));
        x -= 11.0;
    }
    painter.add(egui::Shape::convex_polygon(
        vec![egui::pos2(to, y), egui::pos2(to + 7.0, y - 4.0), egui::pos2(to + 7.0, y + 4.0)],
        border,
        Stroke::NONE,
    ));
    ui.add_space(4.0);
    ui.label(RichText::new(t.t("learn-attention-disclaimer")).small().weak());
}

/// A neutral grid: many learned numbers together, with no individual meaning.
fn tensor_diagram(ui: &mut egui::Ui, t: &I18n) {
    let width = ui.available_width().min(220.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 92.0), Sense::hover());
    let painter = ui.painter();
    let (fill, border, _) = colors(ui);
    let cell = 14.0;
    let gap = 5.0;
    let side = 4.0 * cell + 3.0 * gap;
    let start = egui::pos2(rect.center().x - side / 2.0, rect.top() + 8.0);
    for row in 0..4 {
        for col in 0..4 {
            let cell_rect = egui::Rect::from_min_size(
                egui::pos2(start.x + col as f32 * (cell + gap), start.y + row as f32 * (cell + gap)),
                egui::vec2(cell, cell),
            );
            painter.rect_filled(cell_rect, 2.0, fill);
            painter.rect_stroke(cell_rect, 2.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
        }
    }
    ui.add_space(4.0);
    ui.label(RichText::new(t.t("learn-diagram-tensor")).small().weak());
}

fn colors(ui: &egui::Ui) -> (Color32, Color32, Color32) {
    let v = ui.visuals();
    (v.faint_bg_color, v.widgets.noninteractive.bg_stroke.color, v.text_color())
}

fn arrow_vertical(painter: &egui::Painter, x: f32, from: f32, to: f32, color: Color32) {
    painter.line_segment([egui::pos2(x, from), egui::pos2(x, to - 4.0)], Stroke::new(1.0, color));
    painter.add(egui::Shape::convex_polygon(
        vec![egui::pos2(x - 4.0, to - 5.0), egui::pos2(x + 4.0, to - 5.0), egui::pos2(x, to)],
        color,
        Stroke::NONE,
    ));
}

#[cfg(test)]
#[path = "tests/learn.rs"]
mod tests;
