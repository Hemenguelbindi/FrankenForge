//! Forge: build a new model from a Base and a Donor without a recipe file. The view only collects
//! choices; comparison and planning are ff-surgery use cases run in the background (`jobs`).

pub(crate) mod analysis;
mod build;
mod build_view;
mod jobs;
mod paths;
pub(crate) mod readiness;
mod state;

use std::path::PathBuf;

use eframe::egui::{self, CornerRadius, Margin, RichText, Stroke};
use ff_surgery::compare::TokenizerMatch;

use crate::failure::Failure;
use crate::i18n::I18n;
use crate::loader::Loader;
use crate::views;
use analysis::{Analysis, BlockId, BlockStatus, tokenizer_key};
use build::{BuildRequest, BuildResult};
use build_view::Action;
use state::{Forge, Pair, Preview, Selection};

#[derive(Default)]
pub struct ForgeUi {
    state: Forge,
    analysis: Loader<Pair, Analysis>,
    preview: Loader<Selection, Preview>,
    build: Loader<BuildRequest, BuildResult>,
}

impl ForgeUi {
    /// Applies finished jobs and starts the preview the current selection needs.
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some((req, outcome)) = self.build.poll() {
            self.state.finish_build(req, outcome);
        }
        if let Some((pair, outcome)) = self.analysis.poll() {
            self.state.apply_analysis(pair, outcome);
        }
        if let Some((sel, outcome)) = self.preview.poll() {
            self.state.apply_preview(sel, outcome);
        }
        if let Some(sel) = self.state.preview_needed(self.preview.pending()) {
            let ctx = ctx.clone();
            self.preview.start(sel, jobs::preview, move || ctx.request_repaint());
        }
    }

    fn start_analysis(&mut self, ctx: &egui::Context) {
        self.preview.clear();
        match self.state.pair() {
            Some(pair) => {
                let ctx = ctx.clone();
                self.analysis.start(pair, jobs::analyze, move || ctx.request_repaint());
            }
            None => self.analysis.clear(),
        }
    }
}

/// The whole Forge section, drawn in the central panel. Returns a built model to open in the
/// Inspector.
pub fn show(ui: &mut egui::Ui, t: &I18n, f: &mut ForgeUi) -> Option<PathBuf> {
    egui::CentralPanel::default()
        .show(ui, |ui| {
            f.poll(ui.ctx());
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    ui.heading(t.t("forge-title"));
                    ui.label(RichText::new(t.t("forge-intro")).weak());
                    ui.add_space(12.0);
                    // The state refuses edits during a build too; this only greys the controls out.
                    ui.add_enabled_ui(!f.state.running(), |ui| {
                        sources(ui, t, f);
                        ui.add_space(12.0);
                        compatibility(ui, t, f);
                        ui.add_space(12.0);
                        preview(ui, t, f);
                        ui.add_space(12.0);
                        output(ui, t, f);
                    });
                    ui.add_space(12.0);
                    match build_view::section(ui, t, &mut f.state) {
                        Action::None => None,
                        Action::Open(path) => Some(path),
                        Action::Build => {
                            if let Some(req) = f.state.begin_build() {
                                let ctx = ui.ctx().clone();
                                f.build.start(req, build::run, move || ctx.request_repaint());
                            }
                            None
                        }
                    }
                })
                .inner
        })
        .inner
}

fn sources(ui: &mut egui::Ui, t: &I18n, f: &mut ForgeUi) {
    let base = model_row(ui, t, "forge-base", "forge-base-hint", "forge-dialog-base", f.state.base());
    ui.add_space(8.0);
    let donor = model_row(ui, t, "forge-donor", "forge-donor-hint", "forge-dialog-donor", f.state.donor());
    let changed = base.is_some() || donor.is_some();
    if let Some(path) = base {
        if f.state.out_name().is_empty()
            && let Some(name) = path.file_name()
        {
            f.state.set_out_name(format!("{}-forge", name.to_string_lossy()));
        }
        f.state.set_base(path);
    }
    if let Some(path) = donor {
        f.state.set_donor(path);
    }
    if changed {
        f.start_analysis(ui.ctx());
    }
}

/// A titled folder choice with a one-line explanation; returns a newly picked folder.
fn model_row(
    ui: &mut egui::Ui,
    t: &I18n,
    title: &str,
    hint: &str,
    dialog: &str,
    path: Option<&PathBuf>,
) -> Option<PathBuf> {
    ui.label(RichText::new(t.t(title)).strong().size(16.0));
    ui.label(RichText::new(t.t(hint)).weak());
    ui.horizontal_wrapped(|ui| {
        let picked = ui.button(t.t("forge-choose-folder")).clicked();
        ui.label(
            RichText::new(path.map_or_else(|| t.t("forge-not-chosen"), |p| p.display().to_string()))
                .monospace(),
        );
        picked.then(|| rfd::FileDialog::new().set_title(t.t(dialog)).pick_folder()).flatten()
    })
    .inner
}

fn compatibility(ui: &mut egui::Ui, t: &I18n, f: &mut ForgeUi) {
    ui.label(RichText::new(t.t("forge-compat-title")).strong().size(16.0));
    ui.label(RichText::new(t.t("forge-structural-note")).weak());
    if f.analysis.pending().is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(t.t("forge-analyzing"));
        });
        return;
    }
    let mut toggled = None;
    match f.state.analysis() {
        None => {
            ui.label(t.t("forge-compat-empty"));
        }
        Some(Err(failure)) => failure_box(ui, t, t.t("forge-analysis-failed"), failure, "forge-analysis"),
        Some(Ok(a)) => {
            let status = a.report.tokenizer.status;
            let text = RichText::new(t.t(tokenizer_key(status)));
            if status == TokenizerMatch::Identical {
                ui.label(text);
            } else {
                ui.label(text.color(ui.visuals().error_fg_color));
            }
            for s in &a.report.stacks {
                let rows = a.rows.iter().filter(|r| r.id.stack == s.stack).count();
                let args = [
                    ("stack", s.stack.clone().into()),
                    ("compatible", s.compatible.len().into()),
                    ("total", rows.into()),
                ];
                ui.label(t.ta("forge-stack-summary", &args));
            }
            ui.add_space(6.0);
            toggled = blocks(ui, t, a, &f.state);
        }
    }
    if let Some(id) = toggled {
        f.state.toggle(&id);
    }
}

/// One row per block index of either model; only structurally compatible blocks can be ticked.
fn blocks(ui: &mut egui::Ui, t: &I18n, a: &Analysis, state: &Forge) -> Option<BlockId> {
    let mut toggled = None;
    egui::ScrollArea::vertical().id_salt("forge-blocks").max_height(280.0).show(ui, |ui| {
        for row in &a.rows {
            ui.horizontal(|ui| {
                let mut on = state.is_selected(&row.id);
                let label = t.ta(
                    "forge-block",
                    &[("stack", row.id.stack.clone().into()), ("index", row.id.index.into())],
                );
                let compatible = row.status == BlockStatus::Compatible;
                if ui.add_enabled(compatible, egui::Checkbox::new(&mut on, label)).changed() {
                    toggled = Some(row.id.clone());
                }
                let status = RichText::new(t.t(row.status.message_key()));
                if compatible {
                    ui.label(status.weak());
                } else {
                    let response = ui.label(status.color(ui.visuals().error_fg_color));
                    if let BlockStatus::Different(diffs) = &row.status {
                        response.on_hover_text(diffs.join("\n"));
                    }
                }
            });
        }
    });
    toggled
}

fn preview(ui: &mut egui::Ui, t: &I18n, f: &mut ForgeUi) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t.t("forge-preview-title")).strong().size(16.0));
        if f.state.selection().is_some() && ui.button(t.t("forge-select-none")).clicked() {
            f.state.clear_selection();
        }
    });
    let Some(sel) = f.state.selection() else {
        ui.label(RichText::new(t.t("forge-preview-empty")).weak());
        return;
    };
    let blocks: Vec<String> = sel.blocks.iter().map(|b| format!("{} {}", b.stack, b.index)).collect();
    views::stat(ui, t.t("forge-preview-blocks"), blocks.join(", "));
    match f.state.preview() {
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(t.t("forge-preview-running"));
            });
        }
        Some(Err(failure)) => failure_box(ui, t, t.t("forge-preview-failed"), failure, "forge-preview"),
        Some(Ok(p)) => {
            views::stat(ui, t.t("forge-preview-from-base"), p.from_base.to_string());
            views::stat(ui, t.t("forge-preview-from-donor"), p.from_donor.to_string());
            if !p.notes.is_empty() {
                ui.label(RichText::new(t.t("forge-preview-notes")).weak());
                for n in &p.notes {
                    ui.label(RichText::new(n).monospace().small());
                }
            }
        }
    }
}

fn output(ui: &mut egui::Ui, t: &I18n, f: &mut ForgeUi) {
    ui.label(RichText::new(t.t("forge-output-title")).strong().size(16.0));
    ui.label(RichText::new(t.t("forge-output-hint")).weak());
    ui.horizontal_wrapped(|ui| {
        if ui.button(t.t("forge-output-location")).clicked()
            && let Some(dir) = rfd::FileDialog::new().set_title(t.t("forge-dialog-output")).pick_folder()
        {
            f.state.set_out_dir(dir);
        }
        let dir = f.state.out_dir().map_or_else(|| t.t("forge-not-chosen"), |d| d.display().to_string());
        ui.label(RichText::new(dir).monospace());
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(t.t("forge-output-name"));
        let mut name = f.state.out_name().to_owned();
        if ui.text_edit_singleline(&mut name).changed() {
            f.state.set_out_name(name);
        }
    });
    if let Some(out) = f.state.out() {
        views::stat(ui, t.t("forge-output-path"), out.display().to_string());
    }
}

/// A localized explanation, with the core's original error text under "technical details".
fn failure_box(ui: &mut egui::Ui, t: &I18n, title: String, failure: &Failure, id: &str) {
    let color = ui.visuals().error_fg_color;
    egui::Frame::new()
        .fill(color.gamma_multiply(0.12))
        .stroke(Stroke::new(1.0, color))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).strong().color(color));
            ui.label(t.t(failure.kind.message_key()));
            views::details(ui, t, id, &failure.detail);
        });
}
