//! Forge: build a new model from a Base and a Donor without a recipe file. The view only collects
//! choices; comparison, planning and building are ff-surgery use cases run in the background.
//! Layout, top to bottom: steps, the two sources, compatibility with the block schema, then the
//! composition and Build cards (side by side when the window is wide enough).

pub(crate) mod analysis;
mod build;
pub(crate) mod build_view;
mod composition;
mod jobs;
pub(crate) mod mismatch;
pub(crate) mod palette;
mod paths;
pub(crate) mod readiness;
pub(crate) mod schema;
mod schema_view;
mod state;
pub(crate) mod steps;
pub(crate) mod widgets;

use std::path::PathBuf;

use eframe::egui::{self, RichText};
use ff_surgery::compare::TokenizerMatch;

use crate::i18n::I18n;
use crate::loader::Loader;
use crate::views;
use analysis::{Analysis, tokenizer_key};
use build::{BuildRequest, BuildResult};
use build_view::Action;
use palette::Palette;
use state::{Forge, Pair, Preview, Selection};
use widgets::{card, failure_box, muted, title};

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
    let pal = Palette::of(ui.visuals());
    let frame = egui::Frame::central_panel(ui.style()).fill(pal.surface).inner_margin(18);
    egui::CentralPanel::default()
        .frame(frame)
        .show(ui, |ui| {
            f.poll(ui.ctx());
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| page(ui, t, pal, f)).inner
        })
        .inner
}

fn page(ui: &mut egui::Ui, t: &I18n, pal: Palette, f: &mut ForgeUi) -> Option<PathBuf> {
    ui.label(RichText::new(t.t("forge-title")).size(24.0).strong().color(pal.text));
    muted(ui, pal, t.t("forge-intro"));
    ui.add_space(10.0);
    let problems = f.state.problems();
    card(ui, pal, None, |ui| steps::strip(ui, t, pal, steps::marks(&f.state, &problems)));
    ui.add_space(12.0);
    // The state refuses edits during a build too; this only greys the controls out.
    let running = f.state.running();
    ui.add_enabled_ui(!running, |ui| {
        sources(ui, t, pal, f);
        ui.add_space(12.0);
        compatibility(ui, t, pal, f);
    });
    ui.add_space(12.0);
    let mut action = Action::None;
    widgets::pair(
        ui,
        380.0,
        f,
        |ui, f| {
            ui.add_enabled_ui(!running, |ui| composition::show(ui, t, pal, &mut f.state));
        },
        |ui, f| {
            action = card(ui, pal, Some(pal.seam), |ui| {
                title(ui, pal, t.t("forge-build-title"));
                ui.add_enabled_ui(!running, |ui| output(ui, t, pal, f));
                ui.separator();
                build_view::section(ui, t, pal, &mut f.state)
            });
        },
    );
    match action {
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
}

fn sources(ui: &mut egui::Ui, t: &I18n, pal: Palette, f: &mut ForgeUi) {
    let mut picked = (None, None);
    widgets::pair(
        ui,
        320.0,
        &mut picked,
        |ui, picked| {
            let path = f.state.base();
            picked.0 = model_card(
                ui,
                t,
                pal,
                pal.base,
                ["forge-base", "forge-base-hint", "forge-dialog-base"],
                path,
            );
        },
        |ui, picked| {
            let path = f.state.donor();
            picked.1 = model_card(
                ui,
                t,
                pal,
                pal.donor,
                ["forge-donor", "forge-donor-hint", "forge-dialog-donor"],
                path,
            );
        },
    );
    let changed = picked.0.is_some() || picked.1.is_some();
    if let Some(path) = picked.0 {
        if f.state.out_name().is_empty()
            && let Some(name) = path.file_name()
        {
            f.state.set_out_name(format!("{}-forge", name.to_string_lossy()));
        }
        f.state.set_base(path);
    }
    if let Some(path) = picked.1 {
        f.state.set_donor(path);
    }
    if changed {
        f.start_analysis(ui.ctx());
    }
}

/// A source card: role title, one-line explanation, chosen folder; returns a newly picked folder.
fn model_card(
    ui: &mut egui::Ui,
    t: &I18n,
    pal: Palette,
    accent: egui::Color32,
    [heading, hint, dialog]: [&str; 3],
    path: Option<&PathBuf>,
) -> Option<PathBuf> {
    card(ui, pal, Some(accent), |ui| {
        ui.label(RichText::new(t.t(heading)).size(16.0).strong().color(accent));
        muted(ui, pal, t.t(hint));
        ui.add_space(4.0);
        let shown = path.map_or_else(|| t.t("forge-not-chosen"), |p| p.display().to_string());
        ui.add(egui::Label::new(RichText::new(shown).monospace().color(pal.text)).wrap());
        let picked = ui.button(t.t("forge-choose-folder")).clicked();
        picked.then(|| rfd::FileDialog::new().set_title(t.t(dialog)).pick_folder()).flatten()
    })
}

fn compatibility(ui: &mut egui::Ui, t: &I18n, pal: Palette, f: &mut ForgeUi) {
    let toggled = card(ui, pal, None, |ui| {
        title(ui, pal, t.t("forge-compat-title"));
        muted(ui, pal, t.t("forge-structural-note"));
        ui.add_space(6.0);
        if f.analysis.pending().is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(t.t("forge-analyzing")).color(pal.text));
            });
            return None;
        }
        let a = match f.state.analysis() {
            None => {
                muted(ui, pal, t.t("forge-compat-empty"));
                return None;
            }
            Some(Err(failure)) => {
                failure_box(ui, t, pal, t.t("forge-analysis-failed"), failure, "forge-analysis");
                return None;
            }
            Some(Ok(a)) => a,
        };
        let identical = a.report.tokenizer.status == TokenizerMatch::Identical;
        ui.horizontal_wrapped(|ui| {
            if identical {
                widgets::check(ui, pal);
            }
            let color = if identical { pal.text } else { pal.error };
            ui.label(RichText::new(t.t(tokenizer_key(a.report.tokenizer.status))).color(color));
        });
        for s in &a.report.stacks {
            let rows = a.rows.iter().filter(|r| r.id.stack == s.stack).count();
            let args = [
                ("stack", s.stack.clone().into()),
                ("compatible", s.compatible.len().into()),
                ("total", rows.into()),
            ];
            ui.label(RichText::new(t.ta("forge-stack-summary", &args)).color(pal.text));
        }
        let reasons = mismatch::mismatches(&a.report);
        if !reasons.is_empty() {
            ui.add_space(6.0);
            egui::Frame::new().fill(pal.well).corner_radius(6).inner_margin(10).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(t.t("forge-mismatch-title")).strong().color(pal.text));
                for r in &reasons {
                    ui.label(RichText::new(format!("• {}", r.text(t))).color(pal.text));
                }
            });
        }
        ui.add_space(8.0);
        muted(ui, pal, t.t("forge-schema-hint"));
        let schemas = schema::stacks(a);
        let links = f.state.selection().map(|s| schema::same_index_links(&s.blocks)).unwrap_or_default();
        let clicked = egui::ScrollArea::vertical()
            .id_salt("forge-schema")
            .max_height(320.0)
            .show(ui, |ui| schema_view::show(ui, t, pal, &schemas, &links))
            .inner;
        schema_view::legend(ui, t, pal);
        clicked
    });
    if let Some(id) = toggled {
        f.state.toggle(&id);
    }
}

fn output(ui: &mut egui::Ui, t: &I18n, pal: Palette, f: &mut ForgeUi) {
    muted(ui, pal, t.t("forge-output-hint"));
    ui.horizontal_wrapped(|ui| {
        if ui.button(t.t("forge-output-location")).clicked()
            && let Some(dir) = rfd::FileDialog::new().set_title(t.t("forge-dialog-output")).pick_folder()
        {
            f.state.set_out_dir(dir);
        }
        let dir = f.state.out_dir().map_or_else(|| t.t("forge-not-chosen"), |d| d.display().to_string());
        ui.add(egui::Label::new(RichText::new(dir).monospace().color(pal.text)).wrap());
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t.t("forge-output-name")).color(pal.muted));
        let mut name = f.state.out_name().to_owned();
        if ui.text_edit_singleline(&mut name).changed() {
            f.state.set_out_name(name);
        }
    });
    if let Some(out) = f.state.out() {
        views::stat(ui, t.t("forge-output-path"), out.display().to_string());
    }
}
