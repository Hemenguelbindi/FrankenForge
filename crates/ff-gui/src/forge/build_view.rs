//! The bottom of the Forge page: what still blocks a build, the Build button, the running build
//! and its result.

use std::path::{Path, PathBuf};

use eframe::egui::{self, CornerRadius, Margin, RichText, Stroke};

use super::build::{BuildError, Built};
use super::failure_box;
use super::paths::OutputProblem;
use super::readiness::{Problem, build_enabled};
use super::state::{BuildStatus, Forge};
use crate::i18n::I18n;
use crate::views;

pub enum Action {
    None,
    Build,
    /// Open the built model in the Inspector.
    Open(PathBuf),
}

pub fn section(ui: &mut egui::Ui, t: &I18n, state: &mut Forge) -> Action {
    let problems = state.problems();
    let running = state.running();
    let mut action = Action::None;
    if !running {
        if problems.is_empty() {
            ui.label(RichText::new(t.t("forge-ready")).strong());
        } else {
            ui.label(RichText::new(t.t("forge-build-blocked")).strong());
            for p in &problems {
                ui.label(format!("• {}", problem_text(t, p)));
                if let Problem::Output(OutputProblem::Unresolved(detail)) = p {
                    views::details(ui, t, "forge-output-details", detail);
                }
            }
        }
        ui.add_space(6.0);
    }
    let button = egui::Button::new(RichText::new(t.t("forge-build")).strong());
    if ui.add_enabled(build_enabled(&problems, running), button).clicked() {
        action = Action::Build;
    }
    ui.add_space(6.0);
    match state.build_status() {
        BuildStatus::Idle => {}
        BuildStatus::Running(req) => {
            ui.horizontal_wrapped(|ui| {
                ui.spinner();
                ui.label(t.ta("forge-building", &[("path", req.out.display().to_string().into())]));
            });
            ui.label(RichText::new(t.t("forge-locked")).weak());
        }
        BuildStatus::Done(Ok(built)) => {
            if success(ui, t, built) {
                action = Action::Open(built.out.clone());
            }
        }
        BuildStatus::Done(Err(BuildError::Refused(p))) => {
            let color = ui.visuals().error_fg_color;
            ui.label(RichText::new(t.t("forge-build-refused")).strong().color(color));
            ui.label(problem_text(t, &Problem::Output(p.clone())));
        }
        BuildStatus::Done(Err(BuildError::Failed { failure, recipe, log })) => {
            failure_box(ui, t, t.t("forge-build-failed"), failure, "forge-build-failure");
            if recipe.is_some() || log.is_some() {
                ui.label(RichText::new(t.t("forge-build-failed-kept")).weak());
                for p in recipe.iter().chain(log) {
                    path_row(ui, p);
                }
            }
        }
    }
    action
}

fn problem_text(t: &I18n, p: &Problem) -> String {
    match p {
        Problem::Output(OutputProblem::Exists(path)) => {
            t.ta(p.message_key(), &[("path", path.display().to_string().into())])
        }
        _ => t.t(p.message_key()),
    }
}

/// The verified result with every file written next to it; returns true when "Open" was clicked.
fn success(ui: &mut egui::Ui, t: &I18n, b: &Built) -> bool {
    let color = ui.visuals().hyperlink_color;
    egui::Frame::new()
        .stroke(Stroke::new(1.0, color))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(t.t("forge-build-ok")).strong().color(color));
            ui.label(RichText::new(t.t("forge-build-ok-note")).weak());
            ui.add_space(4.0);
            stat_path(ui, t.t("forge-result-model"), &b.out);
            stat_path(ui, t.t("forge-result-recipe"), &b.recipe);
            stat_path(ui, t.t("forge-result-lock"), &b.lock);
            match &b.log {
                Some(log) => stat_path(ui, t.t("forge-result-log"), log),
                None => {
                    views::stat(ui, t.t("forge-result-log"), t.t("forge-result-log-missing"));
                }
            }
            views::stat(ui, t.t("overview-tensors"), b.tensors.to_string());
            views::stat(ui, t.t("overview-parameters"), t.count(b.params))
                .on_hover_text(b.params.to_string());
            views::stat(ui, t.t("forge-preview-from-base"), b.planned.from_base.to_string());
            views::stat(ui, t.t("forge-preview-from-donor"), b.planned.from_donor.to_string());
            for w in &b.warnings {
                ui.label(RichText::new(w).monospace().small());
            }
            ui.add_space(4.0);
            ui.button(t.t("forge-open-result")).clicked()
        })
        .inner
}

/// Paths are selectable so they can be copied.
fn stat_path(ui: &mut egui::Ui, label: String, path: &Path) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).weak());
        path_row(ui, path);
    });
}

fn path_row(ui: &mut egui::Ui, path: &Path) {
    ui.add(egui::Label::new(RichText::new(path.display().to_string()).monospace()).selectable(true));
}
