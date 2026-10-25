//! Draws the block schema: the Base lane above, the Donor lane below, a seam for every link.
//! Lanes wrap together line by line, so a narrow window never cuts cells off. Every state has its
//! own shape (solid, filled with a check, hatched, dashed, dotted) and a tooltip in words.

use eframe::egui::{self, Align2, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, vec2};

use super::analysis::BlockId;
use super::palette::Palette;
use super::schema::{
    CELL, Cell, GAP, Lane, Link, Placement, Side, StackSchema, Transfer, columns, lines, place,
};
use crate::i18n::I18n;

const LANE: f32 = 30.0;
const SEAM: f32 = 22.0;

/// Returns a Donor block the user clicked; only directly transferable blocks react.
pub fn show(
    ui: &mut egui::Ui,
    t: &I18n,
    pal: Palette,
    schemas: &[StackSchema],
    links: &[Link],
) -> Option<BlockId> {
    let mut clicked = None;
    let names = [t.t("forge-row-base"), t.t("forge-row-donor")];
    let font = FontId::proportional(13.0);
    let widest = names
        .iter()
        .map(|n| ui.painter().layout_no_wrap(n.clone(), font.clone(), pal.text).size().x)
        .fold(0.0, f32::max);
    // Long translations wrap inside the label column instead of eating the lanes.
    let label = (widest + 16.0).min(ui.available_width() * 0.35);
    for s in schemas {
        if schemas.len() > 1 {
            ui.label(RichText::new(&s.stack).monospace().small().color(pal.muted));
        }
        let cols = columns(ui.available_width(), label);
        let own: Vec<&Link> = links.iter().filter(|l| l.stack == s.stack).collect();
        for (li, range) in lines(s.width(), cols).into_iter().enumerate() {
            let (rect, _) =
                ui.allocate_exact_size(vec2(ui.available_width(), LANE * 2.0 + SEAM), Sense::hover());
            let p = ui.painter();
            for (side, name, y) in
                [(Side::Base, &names[0], rect.top()), (Side::Donor, &names[1], rect.top() + LANE + SEAM)]
            {
                let color = side_color(pal, side);
                p.rect_filled(
                    Rect::from_min_size(Pos2::new(rect.left(), y + 6.0), vec2(3.0, LANE - 12.0)),
                    1.0,
                    color,
                );
                let galley = p.layout(name.clone(), font.clone(), pal.text, label - 12.0);
                p.galley(Pos2::new(rect.left() + 8.0, y + (LANE - galley.size().y) / 2.0), galley, pal.text);
            }
            let cell_rect = |pos: usize, side: Side| {
                let x = rect.left() + label + (pos - range.start) as f32 * (CELL + GAP);
                let y = if side == Side::Base { rect.top() } else { rect.top() + LANE + SEAM };
                Rect::from_min_size(Pos2::new(x, y), vec2(CELL, LANE))
            };
            for link in &own {
                if let Placement::Same { line, .. } = place(link, cols)
                    && line == li
                {
                    let (from, to) = (cell_rect(link.donor, Side::Donor), cell_rect(link.base, Side::Base));
                    ui.painter()
                        .line_segment([from.center_top(), to.center_bottom()], Stroke::new(2.0, pal.seam));
                    ui.painter().circle_filled(to.center_bottom() + vec2(0.0, 3.0), 3.0, pal.seam);
                }
            }
            for pos in range.clone() {
                for (lane, side) in [(&s.base, Side::Base), (&s.donor, Side::Donor)] {
                    let Some(cell) = lane.cells.get(pos) else { continue };
                    let r = cell_rect(pos, side);
                    let link =
                        own.iter().find(|l| if side == Side::Base { l.base == pos } else { l.donor == pos });
                    let partner = link.map(|l| if side == Side::Base { l.donor } else { l.base });
                    let id = ui.make_persistent_id(("forge-cell", &s.stack, pos, side == Side::Base));
                    let direct = matches!(cell, Cell::Block { transfer, .. } if transfer.available());
                    let sense = if side == Side::Donor && direct { Sense::click() } else { Sense::hover() };
                    let resp = ui.interact(r, id, sense);
                    let look =
                        Look { linked: partner.is_some(), hot: resp.hovered() && direct, number: true };
                    paint_cell(ui.painter(), pal, side, cell, look, r);
                    // A link whose ends sit on different lines names the other end instead of a seam.
                    if let (Some(l), Some(other)) = (link, partner)
                        && matches!(place(l, cols), Placement::Split { .. })
                    {
                        let at = if side == Side::Base { r.center_bottom() } else { r.center_top() };
                        let align =
                            if side == Side::Base { Align2::CENTER_TOP } else { Align2::CENTER_BOTTOM };
                        ui.painter().text(at, align, other.to_string(), FontId::monospace(9.0), pal.seam);
                    }
                    let tip = tooltip(t, s, lane, cell, partner);
                    if resp.on_hover_text(tip).clicked() {
                        clicked = Some(BlockId { stack: s.stack.clone(), index: pos });
                    }
                }
            }
        }
        ui.add_space(8.0);
    }
    clicked
}

fn side_color(pal: Palette, side: Side) -> egui::Color32 {
    if side == Side::Base { pal.base } else { pal.donor }
}

#[derive(Clone, Copy)]
struct Look {
    /// One end of a link: taken from the Donor, or replaced in the Base.
    linked: bool,
    /// Hovered and clickable.
    hot: bool,
    /// The block index inside the cell (off for legend swatches).
    number: bool,
}

fn paint_cell(p: &egui::Painter, pal: Palette, side: Side, cell: &Cell, look: Look, r: Rect) {
    let Look { linked, hot, number: show } = look;
    let color = side_color(pal, side);
    let number = |c| {
        if show {
            p.text(
                r.center(),
                Align2::CENTER_CENTER,
                cell_index(cell).to_string(),
                FontId::monospace(11.0),
                c,
            );
        }
    };
    match cell {
        Cell::Gap { .. } => {
            for dx in [-6.0, 0.0, 6.0] {
                p.circle_filled(r.center() + vec2(dx, 0.0), 1.5, pal.muted);
            }
        }
        Cell::Block { transfer: Transfer::Direct, .. } if linked && side == Side::Donor => {
            p.rect_filled(r, 4.0, color);
            number(pal.card);
            let c = r.right_top() + vec2(-7.0, 7.0);
            p.add(Shape::line(
                vec![c + vec2(-3.0, 0.0), c + vec2(-1.0, 2.0), c + vec2(3.0, -2.0)],
                Stroke::new(1.5, pal.card),
            ));
        }
        Cell::Block { transfer: Transfer::Direct, .. } if linked => {
            p.rect_filled(r, 4.0, pal.tint(pal.seam));
            p.rect_stroke(r, 4.0, Stroke::new(2.0, pal.seam), StrokeKind::Inside);
            p.rect_filled(
                Rect::from_min_size(r.left_top() + vec2(4.0, 3.0), vec2(CELL - 8.0, 3.0)),
                1.0,
                pal.donor,
            );
            number(pal.text);
        }
        Cell::Block { transfer: Transfer::Direct, .. } => {
            p.rect_filled(r, 4.0, pal.tint(color));
            p.rect_stroke(r, 4.0, Stroke::new(if hot { 2.0 } else { 1.0 }, color), StrokeKind::Inside);
            number(pal.text);
        }
        Cell::Block { transfer: Transfer::Unsupported, .. } => {
            p.rect_filled(r, 4.0, pal.well);
            for k in 0..4 {
                let x = r.left() + 4.0 + k as f32 * 9.0;
                p.line_segment(
                    [Pos2::new(x, r.bottom() - 3.0), Pos2::new(x + 8.0, r.top() + 3.0)],
                    (1.0, pal.tint(pal.error)),
                );
            }
            p.rect_stroke(r, 4.0, Stroke::new(1.0, pal.error), StrokeKind::Inside);
            number(pal.text);
        }
        // Needs a mapping or (in future) an adapter: present, but not transferable as-is.
        Cell::Block { transfer, .. } => {
            let c = if *transfer == Transfer::NeedsAdapter { pal.seam } else { pal.muted };
            let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            p.extend(Shape::dashed_line(&pts, Stroke::new(1.0, c), 4.0, 3.0));
            number(pal.muted);
        }
    }
}

fn cell_index(cell: &Cell) -> usize {
    match cell {
        Cell::Block { index, .. } | Cell::Gap { index } => *index,
    }
}

fn tooltip(t: &I18n, s: &StackSchema, lane: &Lane, cell: &Cell, partner: Option<usize>) -> String {
    let index = cell_index(cell);
    let model = t.t(if lane.side == Side::Base { "forge-row-base" } else { "forge-row-donor" });
    let mut text = format!(
        "{model} · {}",
        t.ta("forge-block", &[("stack", s.stack.clone().into()), ("index", index.into())])
    );
    let state = match (cell, partner, lane.side) {
        (Cell::Gap { .. }, ..) => t.t("forge-cell-gap"),
        (_, Some(other), Side::Base) => t.ta("forge-cell-replaced", &[("index", other.into())]),
        (_, Some(other), Side::Donor) => t.ta("forge-cell-taken", &[("index", other.into())]),
        (Cell::Block { transfer, .. }, None, _) => t.t(transfer.message_key()),
    };
    text.push('\n');
    text.push_str(&state);
    for d in s.differences.get(&index).into_iter().flatten() {
        text.push('\n');
        text.push_str(d);
    }
    text
}

/// One swatch per state, drawn with the same shapes as the cells.
pub fn legend(ui: &mut egui::Ui, t: &I18n, pal: Palette) {
    let block = |transfer| Cell::Block { index: 0, transfer };
    let mut items = vec![(block(Transfer::Direct), true, "forge-legend-taken")];
    items.extend(Transfer::ALL.map(|tr| (block(tr), false, tr.message_key())));
    items.push((Cell::Gap { index: 0 }, false, "forge-cell-gap"));
    ui.horizontal_wrapped(|ui| {
        for (cell, linked, key) in items {
            let side = Side::Donor;
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(CELL, LANE), Sense::hover());
                    paint_cell(ui.painter(), pal, side, &cell, Look { linked, hot: false, number: false }, r);
                    ui.label(RichText::new(t.t(key)).small().color(pal.muted));
                });
            });
            ui.add_space(8.0);
        }
    });
}
