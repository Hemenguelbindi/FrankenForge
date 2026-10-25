//! Drawing for the Tensor Inspector panel. All state and the profile text rules live in the
//! parent module; this file only decides what is visible in Simple and Expert mode.

use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, Sense, Stroke};
use ff_core::TensorInfo;

use super::{InspectorState, Mode, Outcome, Selection, conclusion, number, percent};
use crate::analysis::{Analysis, BINS, Finite, Profile, byte_len};
use crate::failure::Failure;
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::semantics::{Basis, Dim, Param, TensorDoc};
use crate::views::details;

/// Draws the panel; returns true when the user closes it.
pub fn panel(ui: &mut egui::Ui, t: &I18n, inspected: &Inspected, state: &mut InspectorState) -> bool {
    let mut close = false;
    let mut analyze = false;
    let mut mode = state.mode;
    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        let Some(sel) = &state.selection else { return };
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(t.t("inspector-title")).small().weak());
            for m in [Mode::Simple, Mode::Expert] {
                ui.selectable_value(&mut mode, m, t.t(m.message_key()));
            }
            close = ui.small_button(t.t("inspector-close")).clicked();
        });
        ui.add_space(6.0);
        match state.mode {
            Mode::Simple => simple(ui, t, state, sel, &mut analyze),
            Mode::Expert => expert(ui, t, state, sel, &mut analyze),
        }
    });
    state.set_mode(mode);
    if analyze {
        state.start(ui.ctx(), inspected);
    }
    close
}

/// Plain language first: what it is, what it is for, how big it is, and a one-sentence verdict
/// on the profile. Everything numeric stays under "Technical details".
fn simple(ui: &mut egui::Ui, t: &I18n, state: &InspectorState, sel: &Selection, analyze: &mut bool) {
    purpose_simple(ui, t, &sel.doc);
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t.t("overview-parameters")).weak());
        let count = element_count(&sel.info).map_or("-".into(), |n| t.count(n as usize));
        ui.label(RichText::new(count).strong());
    });
    ui.add_space(8.0);
    ui.separator();
    section(ui, t.t("analyze-title"));
    analyze_controls(ui, t, state, sel, analyze);
    if let Some(outcome) = result_for(state, &sel.name) {
        ui.add_space(6.0);
        simple_result(ui, t, outcome);
    }
    ui.add_space(8.0);
    technical_details(ui, t, state, sel);
}

/// Every claim and every number, laid out immediately.
fn expert(ui: &mut egui::Ui, t: &I18n, state: &InspectorState, sel: &Selection, analyze: &mut bool) {
    purpose_expert(ui, t, &sel.doc);
    ui.add_space(8.0);
    section(ui, t.t("inspector-belongs"));
    ui.label(belongs(t, sel));
    ui.add_space(8.0);
    section(ui, t.t("inspector-shape"));
    shape(ui, t, &sel.doc.dims);
    ui.add_space(8.0);
    section(ui, t.t("inspector-technical"));
    technical(ui, t, &sel.info);
    ui.add_space(8.0);
    ui.separator();
    section(ui, t.t("analyze-title"));
    analyze_controls(ui, t, state, sel, analyze);
    if let Some(outcome) = result_for(state, &sel.name) {
        ui.add_space(6.0);
        if let Ok(Analysis::Profile(p)) = outcome
            && let Some(text) = conclusion(t, p)
        {
            ui.label(text);
            ui.add_space(4.0);
        }
        full_result(ui, t, outcome);
    }
}

fn section(ui: &mut egui::Ui, title: String) {
    ui.label(RichText::new(title).strong());
}

fn chip(ui: &mut egui::Ui, t: &I18n, basis: Basis, hint: String) {
    let color = match basis {
        Basis::Confirmed => Color32::from_rgb(74, 170, 110),
        Basis::Inferred => Color32::from_rgb(214, 160, 60),
        Basis::Unknown => Color32::from_rgb(140, 140, 150),
    };
    ui.label(RichText::new(t.t(basis.message_keys().0)).small().color(color)).on_hover_text(hint);
}

fn title(t: &I18n, doc: &TensorDoc) -> String {
    let (title, _) = doc.role.message_keys();
    let mut heading = match doc.expert {
        Some(i) => t.ta(title, &[("index", i.into())]),
        None => t.t(title),
    };
    if doc.param == Param::Bias {
        heading = t.ta("inspector-bias", &[("title", heading.into())]);
    }
    heading
}

fn basis_hint(t: &I18n, doc: &TensorDoc) -> String {
    t.ta(doc.basis.message_keys().1, &[("arch", doc.architecture.clone().unwrap_or_default().into())])
}

fn purpose_simple(ui: &mut egui::Ui, t: &I18n, doc: &TensorDoc) {
    ui.heading(title(t, doc));
    ui.label(t.t(doc.role.message_keys().1));
    // A guess is spelled out in words, not just as a colored status chip.
    if doc.basis == Basis::Inferred {
        ui.label(RichText::new(t.t("inspector-inferred-human")).small().weak());
    }
}

fn purpose_expert(ui: &mut egui::Ui, t: &I18n, doc: &TensorDoc) {
    let hint = basis_hint(t, doc);
    ui.horizontal_wrapped(|ui| {
        ui.heading(title(t, doc));
        chip(ui, t, doc.basis, hint.clone());
    });
    // The basis explanation is part of the answer, not hidden in a tooltip.
    ui.label(RichText::new(hint).small().weak());
    ui.add_space(4.0);
    ui.label(t.t(doc.role.message_keys().1));
}

/// Component card title, then the block (or "outside blocks").
fn belongs(t: &I18n, sel: &Selection) -> String {
    let place = match &sel.block {
        Some((stack, index)) => {
            t.ta("inspector-block", &[("stack", stack.clone().into()), ("index", (*index).into())])
        }
        None => t.t("inspector-outside-blocks"),
    };
    match sel.component {
        Some(kind) => format!("{} · {place}", t.t(kind.message_keys().0)),
        None => place,
    }
}

fn dim_hint_key(basis: Basis) -> &'static str {
    match basis {
        Basis::Confirmed => "dim-basis-confirmed-hint",
        Basis::Inferred => "dim-basis-inferred-hint",
        Basis::Unknown => "dim-basis-unknown-hint",
    }
}

fn shape(ui: &mut egui::Ui, t: &I18n, dims: &[Dim]) {
    if dims.is_empty() {
        ui.label(t.t("inspector-scalar"));
        return;
    }
    for (i, dim) in dims.iter().enumerate() {
        ui.horizontal_wrapped(|ui| {
            let label = t.ta("inspector-dim", &[("index", i.into()), ("size", dim.size.into())]);
            ui.label(RichText::new(label).monospace());
            if let Some(m) = &dim.meaning {
                let mut meaning = t.t(m.key);
                if let Some(f) = &m.formula {
                    meaning = format!("{meaning} ({f})");
                }
                ui.label(meaning);
            }
            chip(ui, t, dim.basis, t.t(dim_hint_key(dim.basis)));
        });
    }
    if dims.iter().all(|d| d.meaning.is_none()) {
        ui.label(RichText::new(t.t("inspector-shape-unknown")).small().weak());
    }
}

fn technical(ui: &mut egui::Ui, t: &I18n, info: &TensorInfo) {
    ui.add(egui::Label::new(RichText::new(&info.name).monospace()).selectable(true));
    egui::Grid::new("technical").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new(t.t("inspector-dtype")).weak());
        ui.monospace(format!("{:?}", info.dtype));
        ui.end_row();
        ui.label(RichText::new(t.t("grid-shape")).weak());
        ui.monospace(format!("{:?}", info.shape));
        ui.end_row();
        ui.label(RichText::new(t.t("inspector-elements")).weak());
        ui.monospace(element_count(info).map_or("-".into(), |n| n.to_string()));
        ui.end_row();
        ui.label(RichText::new(t.t("inspector-size")).weak());
        ui.label(byte_len(info).map_or("-".into(), |n| t.bytes(n)));
        ui.end_row();
    });
}

/// Number of elements from a shape, checked so a damaged header cannot wrap.
fn element_count(info: &TensorInfo) -> Option<u64> {
    info.shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
}

/// Everything factual that Simple mode keeps out of the way: placement, shape meanings, rule
/// bases, the raw identity of the tensor, and the full numeric profile once it exists.
fn technical_details(ui: &mut egui::Ui, t: &I18n, state: &InspectorState, sel: &Selection) {
    egui::CollapsingHeader::new(t.t("inspector-technical")).id_salt("simple-technical-details").show(
        ui,
        |ui| {
            section(ui, t.t("inspector-belongs"));
            ui.label(belongs(t, sel));
            ui.add_space(8.0);
            section(ui, t.t("inspector-shape"));
            shape(ui, t, &sel.doc.dims);
            ui.add_space(8.0);
            section(ui, t.t("inspector-rule-basis"));
            let hint = basis_hint(t, &sel.doc);
            ui.horizontal_wrapped(|ui| {
                chip(ui, t, sel.doc.basis, hint.clone());
                ui.label(RichText::new(hint).small().weak());
            });
            ui.add_space(8.0);
            technical(ui, t, &sel.info);
            if let Some(outcome) = result_for(state, &sel.name) {
                ui.add_space(8.0);
                ui.separator();
                section(ui, t.t("analyze-title"));
                full_result(ui, t, outcome);
            }
        },
    );
}

fn result_for<'a>(state: &'a InspectorState, name: &str) -> Option<&'a Outcome<Analysis>> {
    state.result.as_ref().filter(|(n, _)| n == name).map(|(_, outcome)| outcome)
}

/// Draws the button, or the progress bar while this tensor is being read; sets `analyze` on a
/// click. Nothing is read until then.
fn analyze_controls(
    ui: &mut egui::Ui,
    t: &I18n,
    state: &InspectorState,
    sel: &Selection,
    analyze: &mut bool,
) {
    if state.loader.pending() == Some(&sel.name) {
        let done = state.progress.load(Ordering::Relaxed);
        let fraction = if state.total == 0 { 0.0 } else { (done as f64 / state.total as f64) as f32 };
        let percent = format!("{:.0}", fraction * 100.0);
        ui.add(
            egui::ProgressBar::new(fraction).text(t.ta("analyze-running", &[("percent", percent.into())])),
        );
        // The worker wakes the UI only when done; keep the bar moving meanwhile.
        ui.ctx().request_repaint_after(Duration::from_millis(100));
        return;
    }
    ui.label(RichText::new(t.t("analyze-hint")).small().weak());
    *analyze = ui.button(t.t("analyze-button")).clicked();
}

/// A short conclusion for Simple mode. It states the facts of the profile and never calls the
/// tensor or the model healthy, normal or knowledgeable.
fn simple_result(ui: &mut egui::Ui, t: &I18n, outcome: &Outcome<Analysis>) {
    match outcome {
        Ok(Analysis::Profile(p)) => {
            ui.label(conclusion(t, p).unwrap_or_else(|| t.t("analyze-empty")));
        }
        Ok(Analysis::Empty) => {
            ui.label(t.t("analyze-empty"));
        }
        Ok(Analysis::Unsupported(dtype)) => {
            ui.label(t.ta("analyze-unsupported", &[("dtype", format!("{dtype:?}").into())]));
        }
        Err(failure) => analysis_error(ui, t, failure),
    }
}

/// The full numbers for Expert mode and for Simple mode's expanded details.
fn full_result(ui: &mut egui::Ui, t: &I18n, outcome: &Outcome<Analysis>) {
    match outcome {
        Ok(Analysis::Profile(p)) => profile(ui, t, p),
        Ok(Analysis::Empty) => {
            ui.label(t.t("analyze-empty"));
        }
        Ok(Analysis::Unsupported(dtype)) => {
            ui.label(t.ta("analyze-unsupported", &[("dtype", format!("{dtype:?}").into())]));
        }
        Err(failure) => analysis_error(ui, t, failure),
    }
}

fn analysis_error(ui: &mut egui::Ui, t: &I18n, failure: &Failure) {
    ui.colored_label(ui.visuals().error_fg_color, t.t("analyze-failed"));
    ui.label(t.t(failure.kind.message_key()));
    details(ui, t, "analysis-error-details", &failure.detail);
}

fn profile(ui: &mut egui::Ui, t: &I18n, p: &Profile) {
    let row = |ui: &mut egui::Ui, key: &str, value: String| {
        ui.label(RichText::new(t.t(key)).weak()).on_hover_text(t.t(&format!("{key}-hint")));
        ui.monospace(value);
        ui.end_row();
    };
    egui::Grid::new("profile").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
        row(ui, "stat-count", p.count.to_string());
        if let Some(f) = &p.finite {
            row(ui, "stat-min", number(t, f.min));
            row(ui, "stat-max", number(t, f.max));
            row(ui, "stat-mean", number(t, f.mean));
            row(ui, "stat-std", number(t, f.std));
        }
        row(ui, "stat-zeros", format!("{}% ({})", percent(t, p.zeros, p.count), p.zeros));
        row(ui, "stat-nan", p.nan.to_string());
        row(ui, "stat-inf", format!("+{} / -{}", p.pos_inf, p.neg_inf));
    });
    if p.nan + p.pos_inf + p.neg_inf > 0 {
        ui.colored_label(ui.visuals().warn_fg_color, t.t("stat-nonfinite-warning"));
    }
    match &p.finite {
        Some(f) => histogram(ui, t, f),
        None => {
            ui.label(t.t("stat-no-finite"));
        }
    }
    ui.add_space(6.0);
    ui.label(RichText::new(t.t("stat-explainer")).small().weak());
}

fn histogram(ui: &mut egui::Ui, t: &I18n, f: &Finite) {
    ui.add_space(6.0);
    ui.label(RichText::new(t.t("stat-histogram")).weak());
    let width = ui.available_width().max(120.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 110.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let (border, fill) = {
        let v = ui.visuals();
        (v.widgets.noninteractive.bg_stroke.color, v.hyperlink_color.gamma_multiply(0.8))
    };
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
    let peak = f.histogram.iter().copied().max().unwrap_or(0).max(1) as f32;
    let bar_w = rect.width() / BINS as f32;
    // Values come from 16/32-bit floats, so this width cannot overflow f64.
    let bin_width = (f.max - f.min) / BINS as f64;
    for (i, &n) in f.histogram.iter().enumerate() {
        let h = (n as f32 / peak) * (rect.height() - 6.0);
        let x = rect.left() + i as f32 * bar_w;
        let bar = egui::Rect::from_min_max(
            egui::pos2(x + 1.0, rect.bottom() - h),
            egui::pos2(x + bar_w - 1.0, rect.bottom()),
        );
        painter.rect_filled(bar, 1.0, fill);
        if let Some(pos) = response.hover_pos()
            && pos.x >= x
            && pos.x < x + bar_w
        {
            let lo = f.min + bin_width * i as f64;
            let text = t.ta(
                "stat-bin",
                &[
                    ("from", number(t, lo).into()),
                    ("to", number(t, lo + bin_width).into()),
                    ("count", n.into()),
                ],
            );
            response.clone().on_hover_text_at_pointer(text);
        }
    }
    ui.horizontal(|ui| {
        ui.monospace(number(t, f.min));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.monospace(number(t, f.max));
        });
    });
}
