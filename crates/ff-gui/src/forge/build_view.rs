//! The Build card: what still blocks a build, the Build button, the running build and its result.
//! The core reports no stages, so a running build shows one honest status line; the four stages
//! are listed only after success, when `build::build` guarantees all of them happened.

use std::path::{Path, PathBuf};

use eframe::egui::{self, RichText, Stroke};
use ff_preflight::{Known, PreflightStatus};

use super::build::{BuildError, Built};
use super::palette::Palette;
use super::paths::OutputProblem;
use super::readiness::{Problem, build_enabled};
use super::state::{BuildStatus, Forge};
use super::widgets::{check, failure_box, muted};
use crate::i18n::I18n;
use crate::views;

pub enum Action {
    None,
    Build,
    /// Open the built model in the Inspector.
    Open(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Idle,
    Running,
    Succeeded,
    /// The fresh check before writing refused; nothing was written.
    NotStarted,
    /// The core failed or the worker crashed.
    Stopped,
}

pub fn status(b: &BuildStatus) -> Status {
    match b {
        BuildStatus::Idle => Status::Idle,
        BuildStatus::Running(_) => Status::Running,
        BuildStatus::Done(Ok(_)) => Status::Succeeded,
        BuildStatus::Done(Err(BuildError::Refused(_))) => Status::NotStarted,
        BuildStatus::Done(Err(BuildError::Failed { .. })) => Status::Stopped,
    }
}

pub const STAGES: [&str; 4] =
    ["forge-stage-plan", "forge-stage-copy", "forge-stage-verify", "forge-stage-lock"];

/// Stages are claimed only when the build succeeded; a failure never marks a stage as done.
pub fn completed_stages(s: Status) -> &'static [&'static str] {
    if s == Status::Succeeded { &STAGES } else { &[] }
}

pub fn section(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &mut Forge) -> Action {
    let problems = state.problems();
    let running = state.running();
    let stages = completed_stages(status(state.build_status()));
    let mut action = Action::None;
    if !running {
        if problems.is_empty() {
            ui.label(RichText::new(t.t("forge-ready")).strong().color(pal.ok));
        } else {
            ui.label(RichText::new(t.t("forge-build-blocked")).strong().color(pal.text));
            for p in &problems {
                ui.label(RichText::new(format!("• {}", problem_text(t, p))).color(pal.muted));
                if let Problem::Output(OutputProblem::Unresolved(detail)) = p {
                    views::details(ui, t, "forge-output-details", detail);
                }
            }
        }
        ui.add_space(8.0);
    }
    disk_line(ui, t, pal, state);
    let enabled = build_enabled(&problems, running);
    let label = RichText::new(t.t("forge-build")).strong().size(15.0);
    let button = if enabled {
        egui::Button::new(label.color(pal.card)).fill(pal.seam).min_size(egui::vec2(140.0, 32.0))
    } else {
        egui::Button::new(label).min_size(egui::vec2(140.0, 32.0))
    };
    if ui.add_enabled(enabled, button).clicked() {
        action = Action::Build;
    }
    ui.add_space(8.0);
    match state.build_status() {
        BuildStatus::Idle => {}
        BuildStatus::Running(req) => {
            ui.horizontal_wrapped(|ui| {
                ui.spinner();
                let path = req.out.display().to_string();
                ui.label(
                    RichText::new(t.ta("forge-building", &[("path", path.into())])).strong().color(pal.text),
                );
            });
            muted(ui, pal, t.t("forge-building-includes"));
            muted(ui, pal, t.t("forge-locked"));
        }
        BuildStatus::Done(Ok(built)) => {
            if success(ui, t, pal, built, stages) {
                action = Action::Open(built.out.clone());
            }
        }
        BuildStatus::Done(Err(BuildError::Refused(p))) => {
            ui.label(RichText::new(t.t("forge-build-refused")).strong().color(pal.error));
            ui.label(RichText::new(problem_text(t, &Problem::Output(p.clone()))).color(pal.text));
        }
        BuildStatus::Done(Err(BuildError::Failed { failure, recipe, log })) => {
            failure_box(ui, t, pal, t.t("forge-build-failed"), failure, "forge-build-failure");
            if recipe.is_some() || log.is_some() {
                muted(ui, pal, t.t("forge-build-failed-kept"));
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
        Problem::Output(OutputProblem::NoSpace { need, available }) => t.ta(
            p.message_key(),
            &[("need", t.bytes(*need).into()), ("available", t.bytes(*available).into())],
        ),
        _ => t.t(p.message_key()),
    }
}

/// Hardware Preflight for the build's disk, from cached facts; Unknown is said, never hidden.
fn disk_line(ui: &mut egui::Ui, t: &I18n, pal: Palette, state: &Forge) {
    let Some((status, need, available)) = state.disk() else { return };
    let (color, word) = match status {
        PreflightStatus::Fits => (pal.ok, "pf-status-fits"),
        PreflightStatus::Tight => (pal.donor, "pf-status-tight"),
        PreflightStatus::Insufficient => (pal.error, "pf-status-insufficient"),
        PreflightStatus::Unknown => (pal.muted, "pf-status-unknown"),
    };
    let text = match &available {
        Known::Known(free) => {
            t.ta("forge-disk-line", &[("need", t.bytes(need).into()), ("available", t.bytes(*free).into())])
        }
        Known::Unknown(why) => {
            t.ta("forge-disk-unknown", &[("need", t.bytes(need).into()), ("why", why.clone().into())])
        }
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t.t(word)).strong().color(color));
        ui.label(RichText::new(text).color(pal.text));
    });
}

/// The verified result with every file written next to it; returns true when "Open" was clicked.
fn success(ui: &mut egui::Ui, t: &I18n, pal: Palette, b: &Built, stages: &[&str]) -> bool {
    egui::Frame::new()
        .fill(pal.tint(pal.ok))
        .stroke(Stroke::new(1.0, pal.ok))
        .corner_radius(6)
        .inner_margin(10)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(t.t("forge-build-ok")).strong().size(15.0).color(pal.ok));
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(t.t("forge-stages-done")).color(pal.muted));
                for key in stages {
                    check(ui, pal);
                    ui.label(RichText::new(t.t(key)).color(pal.text));
                }
            });
            muted(ui, pal, t.t("forge-build-ok-note"));
            ui.add_space(4.0);
            stat_path(ui, pal, t.t("forge-result-model"), &b.out);
            stat_path(ui, pal, t.t("forge-result-recipe"), &b.recipe);
            stat_path(ui, pal, t.t("forge-result-lock"), &b.lock);
            match &b.log {
                Some(log) => stat_path(ui, pal, t.t("forge-result-log"), log),
                None => {
                    views::stat(ui, t.t("forge-result-log"), t.t("forge-result-log-missing"));
                }
            }
            views::stat(ui, t.t("overview-tensors"), b.tensors.to_string());
            views::stat(ui, t.t("overview-parameters"), t.count(b.params))
                .on_hover_text(b.params.to_string());
            for w in &b.warnings {
                ui.label(RichText::new(w).monospace().small());
            }
            ui.add_space(6.0);
            ui.button(RichText::new(t.t("forge-open-result")).strong()).clicked()
        })
        .inner
}

/// Paths are selectable so they can be copied.
fn stat_path(ui: &mut egui::Ui, pal: Palette, label: String, path: &Path) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).color(pal.muted));
        path_row(ui, path);
    });
}

fn path_row(ui: &mut egui::Ui, path: &Path) {
    ui.add(egui::Label::new(RichText::new(path.display().to_string()).monospace()).selectable(true));
}

#[cfg(test)]
#[path = "tests/build_view.rs"]
mod tests;
