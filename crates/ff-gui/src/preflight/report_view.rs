//! The estimate card: overall status, one line per resource, what to reduce, and the parts and
//! assumptions behind the numbers. Pure over cached facts: no hardware or disk access.

use eframe::egui::{self, Align2, FontId, RichText, Sense, Stroke, vec2};
use ff_preflight::{Known, PreflightStatus, Span, preflight};

use super::PreflightUi;
use super::state::{advice_key, assumption_key, part_key, resource_key, status_keys};
use crate::forge::palette::Palette;
use crate::forge::widgets::{card, muted, title};
use crate::i18n::I18n;

pub fn span_text(t: &I18n, s: &Known<Span>) -> String {
    match s {
        Known::Known(s) if s.low == s.high => t.bytes(s.low),
        Known::Known(s) => format!("{} – {}", t.bytes(s.low), t.bytes(s.high)),
        Known::Unknown(why) => t.ta("pf-unknown-why", &[("why", why.clone().into())]),
    }
}

pub fn show(ui: &mut egui::Ui, t: &I18n, pal: Palette, p: &PreflightUi) {
    card(ui, pal, Some(pal.seam), |ui| {
        title(ui, pal, t.t("pf-result-title"));
        let (Some(snap), Some(Ok(m))) = (&p.snapshot, &p.model) else {
            muted(ui, pal, t.t("pf-result-need"));
            return;
        };
        let report = preflight(&p.settings.request(m), snap);
        let (word, desc) = status_keys(report.status);
        ui.horizontal_wrapped(|ui| {
            badge(ui, pal, report.status);
            ui.label(RichText::new(t.t(word)).strong().size(16.0).color(status_color(pal, report.status)));
        });
        muted(ui, pal, t.t(desc));
        ui.add_space(6.0);
        for c in &report.checks {
            let mut need = span_text(t, &Known::Known(c.need.span));
            if !c.need.unknown.is_empty() {
                let parts: Vec<String> = c.need.unknown.iter().map(|p| t.t(part_key(*p))).collect();
                need = t
                    .ta("pf-need-plus-unknown", &[("need", need.into()), ("parts", parts.join(", ").into())]);
            }
            let available = span_text(t, &c.available.clone().map(Span::exact));
            ui.horizontal_wrapped(|ui| {
                badge(ui, pal, c.status);
                ui.label(RichText::new(t.t(resource_key(c.resource))).strong().color(pal.text));
                let line = t.ta("pf-check-line", &[("need", need.into()), ("available", available.into())]);
                ui.label(RichText::new(line).color(pal.text));
                ui.label(RichText::new(t.t(status_keys(c.status).0)).color(status_color(pal, c.status)));
            });
        }
        if !report.advice.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new(t.t("pf-advice-title")).strong().color(pal.text));
            for a in &report.advice {
                ui.label(RichText::new(format!("• {}", t.t(advice_key(*a)))).color(pal.text));
            }
        }
        egui::CollapsingHeader::new(t.t("pf-parts-title")).id_salt("pf-parts").show(ui, |ui| {
            for item in &report.estimate.items {
                let part = format!("{} · {}", t.t(part_key(item.part)), t.t(resource_key(item.resource)));
                ui.label(RichText::new(format!("{part}: {}", span_text(t, &item.size))).color(pal.text));
            }
        });
        egui::CollapsingHeader::new(t.t("pf-assumptions-title")).id_salt("pf-assumptions").show(ui, |ui| {
            for a in &report.estimate.assumptions {
                ui.label(RichText::new(format!("• {}", t.t(assumption_key(*a)))).color(pal.text));
            }
        });
        muted(ui, pal, t.t("pf-disclaimer"));
    });
}

fn status_color(pal: Palette, s: PreflightStatus) -> egui::Color32 {
    match s {
        PreflightStatus::Fits => pal.ok,
        PreflightStatus::Tight => pal.donor,
        PreflightStatus::Insufficient => pal.error,
        PreflightStatus::Unknown => pal.muted,
    }
}

/// Shape per status so it never depends on color: filled disc, half disc, "!" ring, "?" ring.
fn badge(ui: &mut egui::Ui, pal: Palette, s: PreflightStatus) {
    let (r, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
    let (c, color, p) = (r.center(), status_color(pal, s), ui.painter());
    match s {
        PreflightStatus::Fits => {
            p.circle_filled(c, 7.0, color);
        }
        PreflightStatus::Tight => {
            p.circle_stroke(c, 7.0, Stroke::new(1.5, color));
            let lower = egui::Rect::from_min_max(c - vec2(7.0, 0.0), c + vec2(7.0, 7.0)).intersect(r);
            p.rect_filled(lower, 3.0, color);
        }
        PreflightStatus::Insufficient | PreflightStatus::Unknown => {
            p.circle_stroke(c, 7.0, Stroke::new(1.5, color));
            let mark = if s == PreflightStatus::Unknown { "?" } else { "!" };
            p.text(c, Align2::CENTER_CENTER, mark, FontId::proportional(11.0), color);
        }
    }
}
