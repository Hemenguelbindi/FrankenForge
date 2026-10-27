use std::path::PathBuf;

use eframe::egui::{self, CornerRadius, Margin, RichText, Sense, Stroke, Theme};
use ff_io::Safetensors;

use crate::components::{Component, Kind};
use crate::failure::Failure;
use crate::forge::{self, ForgeUi};
use crate::graph::{self, canvas::GraphUi};
use crate::i18n::{I18n, Lang, initial_lang, os_locale};
use crate::inspected::Inspected;
use crate::inspector::{self, InspectorState, Mode};
use crate::learn::{self, LearnState};
use crate::loader::Loader;
use crate::preflight::{self, PreflightUi};
use crate::presentation::{Presentation, initial_presentation};
use crate::structure;
use crate::views::{self, component_tree, config, details, hits};
use crate::workspace::{self, Workspace};

/// Storage key for the chosen interface language (a `Lang::code`).
const LANGUAGE_KEY: &str = "language";
/// Storage key for the inspector mode (a `Mode::code`).
const MODE_KEY: &str = "inspector-mode";
/// Storage key for the chosen presentation (a `Presentation::code`).
const PRESENTATION_KEY: &str = "presentation";

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum View {
    #[default]
    Component,
    Config,
}

struct OpenError {
    path: PathBuf,
    failure: Failure,
}

pub struct App {
    t: I18n,
    /// False while the language comes from `--lang`: a one-off override must not replace the
    /// saved choice. Picking a language in the UI makes it the saved choice again.
    persist_lang: bool,
    workspace: Workspace,
    forge: ForgeUi,
    preflight: PreflightUi,
    inspected: Option<Inspected>,
    loader: Loader<PathBuf, Inspected>,
    error: Option<OpenError>,
    selected: Option<Kind>,
    view: View,
    presentation: Presentation,
    learn: LearnState,
    /// The tensor selected in the structure or technical tree; kept across presentation switches.
    picked: Option<String>,
    query: String,
    /// Search result for `query`, recomputed only when the query or the model changes.
    hits: Vec<String>,
    /// The selected tensor and its analysis; weights are read only after "Analyze tensor".
    inspector: InspectorState,
    /// The model graph of the Structure presentation and its "what is inside" questions.
    graph: GraphUi,
}

impl App {
    /// `forced` comes from `--lang`; otherwise the saved choice, then the OS language, then English.
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>, forced: Option<Lang>) -> Self {
        let saved = cc.storage.and_then(|s| s.get_string(LANGUAGE_KEY));
        let lang = forced.unwrap_or_else(|| initial_lang(saved.as_deref(), os_locale().as_deref()));
        // Simple is the default; an unknown saved value falls back to it.
        let mode = cc
            .storage
            .and_then(|s| s.get_string(MODE_KEY))
            .and_then(|v| Mode::from_code(&v))
            .unwrap_or_default();
        let saved_presentation = cc.storage.and_then(|s| s.get_string(PRESENTATION_KEY));
        let presentation = initial_presentation(saved_presentation.as_deref());
        let mut app = Self {
            t: I18n::new(lang),
            persist_lang: forced.is_none(),
            workspace: Workspace::default(),
            forge: ForgeUi::default(),
            preflight: PreflightUi::default(),
            inspected: None,
            loader: Loader::default(),
            error: None,
            selected: None,
            view: View::default(),
            presentation,
            learn: LearnState::default(),
            picked: None,
            query: String::new(),
            hits: Vec::new(),
            inspector: InspectorState::default(),
            graph: GraphUi::default(),
        };
        app.inspector.set_mode(mode);
        if let Some(path) = path {
            app.open(&cc.egui_ctx, path);
        }
        app
    }

    /// Opens in the background. The model on screen stays until the new one is ready; a newer
    /// request supersedes this one and its result is discarded.
    fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        let ctx = ctx.clone();
        self.loader.start(
            path,
            |p| Inspected::open(p, &Safetensors).map_err(|e| Failure::from_error(&e)),
            move || ctx.request_repaint(),
        );
    }

    fn apply_finished(&mut self) {
        let Some((path, outcome)) = self.loader.poll() else { return };
        match outcome {
            Ok(inspected) => {
                self.selected = inspected.components.first().map(|c| c.kind);
                self.view = View::Component;
                self.hits = inspected.search(&self.query);
                self.inspected = Some(inspected);
                self.error = None;
                self.picked = None;
                self.inspector.reset();
            }
            Err(failure) => self.error = Some(OpenError { path, failure }),
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        // Wrapping instead of a fixed row: longer translations move to the next line rather than
        // overlapping or being cut off.
        ui.horizontal_wrapped(|ui| {
            if workspace::switcher(ui, &self.t, &mut self.workspace) {
                self.inspector.stop_analysis();
            }
            ui.separator();
            if ui.button(self.t.t("toolbar-open-file")).clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .set_title(self.t.t("dialog-open-file"))
                    .add_filter("safetensors", &["safetensors"])
                    .pick_file()
            {
                self.open(ui.ctx(), path);
            }
            if ui.button(self.t.t("toolbar-open-folder")).clicked()
                && let Some(path) =
                    rfd::FileDialog::new().set_title(self.t.t("dialog-open-folder")).pick_folder()
            {
                self.open(ui.ctx(), path);
            }
            ui.separator();
            self.language_picker(ui);
            theme_switch(ui, &self.t);
            if let Some(path) = self.loader.pending() {
                ui.separator();
                ui.spinner();
                ui.label(self.t.ta("toolbar-opening", &[("path", path.display().to_string().into())]));
            }
        });
        if let Some(error) = &self.error {
            ui.add_space(4.0);
            if error_banner(ui, &self.t, error) {
                self.error = None;
            }
        }
        if self.workspace == Workspace::Inspector && self.inspected.is_some() {
            self.presentation_tabs(ui);
        }
        ui.add_space(6.0);
    }

    /// The three top-level presentations. Labels are full words, not icons, so a new reader can
    /// find the tour again at any time.
    fn presentation_tabs(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for p in Presentation::ALL {
                let label = RichText::new(self.t.t(p.message_key())).strong();
                if ui.selectable_label(self.presentation == p, label).clicked() {
                    // Streaming a whole tensor that nobody can watch would be wasted disk I/O.
                    if self.presentation == Presentation::Technical && p != Presentation::Technical {
                        self.inspector.stop_analysis();
                    }
                    self.presentation = p;
                }
            }
        });
    }

    fn language_picker(&mut self, ui: &mut egui::Ui) {
        let mut lang = self.t.lang();
        egui::ComboBox::from_id_salt("language")
            .selected_text(lang.native_name())
            .show_ui(ui, |ui| {
                for l in Lang::ALL {
                    ui.selectable_value(&mut lang, l, l.native_name());
                }
            })
            .response
            .on_hover_text(self.t.t("toolbar-language"));
        if lang != self.t.lang() {
            // Applied immediately: every string is looked up again on the next frame.
            self.t.set_lang(lang);
            self.persist_lang = true;
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.persist_lang {
            storage.set_string(LANGUAGE_KEY, self.t.lang().code().to_owned());
        }
        storage.set_string(MODE_KEY, self.inspector.mode().code().to_owned());
        storage.set_string(PRESENTATION_KEY, self.presentation.code().to_owned());
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.apply_finished();
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));

        if self.workspace == Workspace::Preflight {
            return preflight::show(ui, &self.t, &mut self.preflight);
        }
        if self.workspace == Workspace::Forge {
            if let Some(path) = forge::show(ui, &self.t, &mut self.forge) {
                self.workspace = Workspace::Inspector;
                self.open(ui.ctx(), path);
            }
            return;
        }
        let t = &self.t;
        let Some(inspected) = &self.inspected else {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 3.0);
                    ui.heading(t.t("welcome-title"));
                    ui.add_space(6.0);
                    ui.label(t.t("welcome-body"));
                });
            });
            return;
        };

        if self.presentation == Presentation::Learn {
            egui::CentralPanel::default().show(ui, |ui| {
                learn::view(ui, t, inspected, &mut self.learn);
            });
            return;
        }

        if self.presentation == Presentation::Structure {
            egui::Panel::left("overview").resizable(true).default_size(360.0).min_size(260.0).show(
                ui,
                |ui| {
                    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                        ui.add_space(8.0);
                        structure::overview(ui, t, inspected, &mut self.selected, &mut self.picked);
                    });
                },
            );

            self.graph.sync(inspected);
            let mut open_technical = false;
            egui::Panel::right("inside").resizable(true).default_size(440.0).min_size(320.0).show(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                    ui.add_space(8.0);
                    graph::inside::show(
                        ui,
                        t,
                        inspected,
                        &mut self.graph,
                        &mut self.picked,
                        &mut open_technical,
                    );
                });
            });
            egui::CentralPanel::default().show(ui, |ui| graph::canvas::show(ui, t, &mut self.graph));
            // The chosen part is ready for the inspector if the user moves to Technical data.
            self.inspector.select(inspected, self.picked.clone());
            if open_technical {
                self.presentation = Presentation::Technical;
                self.view = View::Component;
            }
            return;
        }

        egui::Panel::left("overview").resizable(true).default_size(360.0).min_size(260.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                ui.add_space(8.0);
                overview(ui, t, inspected, &mut self.selected, &mut self.view);
            });
        });

        self.inspector.poll();
        if self.inspector.selected().is_some() {
            let mut close = false;
            egui::Panel::right("inspector").resizable(true).default_size(400.0).min_size(300.0).show(
                ui,
                |ui| {
                    close = inspector::panel(ui, t, inspected, &mut self.inspector);
                },
            );
            if close {
                self.inspector.reset();
                self.picked = None;
            }
        }

        let mut picked = self.picked.clone();
        egui::CentralPanel::default().show(ui, |ui| {
            let search = egui::TextEdit::singleline(&mut self.query)
                .hint_text(t.t("search-hint"))
                .desired_width(f32::INFINITY);
            if ui.add(search).changed() {
                self.hits = inspected.search(&self.query);
            }
            ui.add_space(6.0);
            if !self.query.trim().is_empty() {
                hits(ui, t, inspected, &self.hits, &mut picked);
                return;
            }
            match self.view {
                View::Config => config(ui, t, &inspected.config),
                View::Component => {
                    match self.selected.and_then(|k| inspected.components.iter().find(|c| c.kind == k)) {
                        Some(c) => component_tree(ui, t, inspected, c, &mut picked),
                        None => {
                            ui.label(t.t("tree-select"));
                        }
                    }
                }
            }
        });
        self.picked = picked;
        // Selecting another tensor drops the previous one's analysis, finished or running.
        self.inspector.select(inspected, self.picked.clone());
    }
}

fn theme_switch(ui: &mut egui::Ui, t: &I18n) {
    let current = ui.ctx().theme();
    let mut theme = current;
    ui.selectable_value(&mut theme, Theme::Dark, t.t("toolbar-theme-dark"))
        .on_hover_text(t.t("toolbar-theme"));
    ui.selectable_value(&mut theme, Theme::Light, t.t("toolbar-theme-light"))
        .on_hover_text(t.t("toolbar-theme"));
    if theme != current {
        ui.ctx().set_theme(theme);
    }
}

/// A localized explanation first; the core's own error text is kept verbatim under "details".
/// Returns true when the user dismissed it.
fn error_banner(ui: &mut egui::Ui, t: &I18n, error: &OpenError) -> bool {
    let color = ui.visuals().error_fg_color;
    let mut dismiss = false;
    egui::Frame::new()
        .fill(color.gamma_multiply(0.12))
        .stroke(Stroke::new(1.0, color))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let title = t.ta("error-open-title", &[("path", error.path.display().to_string().into())]);
            ui.label(RichText::new(title).strong().color(color));
            ui.label(t.t(error.failure.kind.message_key()));
            details(ui, t, "open-error-details", &error.failure.detail);
            dismiss = ui.button(t.t("error-dismiss")).clicked();
        });
    dismiss
}

fn overview(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    selected: &mut Option<Kind>,
    view: &mut View,
) {
    ui.label(RichText::new(t.t("overview-title")).small().weak());
    ui.heading(&inspected.name);
    ui.label(RichText::new(inspected.path.display().to_string()).small().weak());
    ui.add_space(8.0);

    // Wrapped rows instead of a grid: a long label or value wraps inside the panel.
    let dtypes: Vec<String> = inspected.dtypes.iter().map(|(d, n)| format!("{d:?} ({n})")).collect();
    views::stat(ui, t.t("overview-parameters"), t.count(inspected.params))
        .on_hover_text(inspected.params.to_string());
    views::stat(ui, t.t("overview-tensors"), inspected.model.tensors.len().to_string());
    views::stat(ui, t.t("overview-weight-types"), dtypes.join(", "));

    ui.add_space(12.0);
    ui.label(RichText::new(t.t("overview-components")).strong());
    for c in &inspected.components {
        let is_selected = *view == View::Component && *selected == Some(c.kind);
        if card(ui, t, c, is_selected).clicked() {
            *selected = Some(c.kind);
            *view = View::Component;
        }
    }

    ui.add_space(12.0);
    if ui.selectable_label(*view == View::Config, t.t("overview-config")).clicked() {
        *view = View::Config;
    }
}

fn card(ui: &mut egui::Ui, t: &I18n, c: &Component, selected: bool) -> egui::Response {
    let (stroke, fill) = {
        let v = ui.visuals();
        if selected {
            (Stroke::new(2.0, v.hyperlink_color), v.selection.bg_fill.gamma_multiply(0.35))
        } else {
            (v.widgets.noninteractive.bg_stroke, v.faint_bg_color)
        }
    };
    let (title, purpose) = c.kind.message_keys();
    let params = t.count(c.params);
    let size = if c.block_count() > 0 {
        t.ta("card-blocks", &[("blocks", c.block_count().into()), ("params", params.into())])
    } else {
        t.ta("card-tensors", &[("tensors", c.loose.len().into()), ("params", params.into())])
    };
    let response = egui::Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            // Otherwise clicks on the text would select it instead of opening the card.
            ui.style_mut().interaction.selectable_labels = false;
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(t.t(title)).strong().size(16.0));
                ui.label(RichText::new(size).weak());
            });
            ui.label(t.t(purpose));
            for s in &c.stacks {
                let line = t
                    .ta("stack-blocks", &[("name", s.name.clone().into()), ("count", s.blocks.len().into())]);
                ui.label(RichText::new(line).monospace().small().weak());
            }
            if !c.loose.is_empty() && !c.stacks.is_empty() {
                ui.label(
                    RichText::new(t.ta("card-loose", &[("count", c.loose.len().into())])).small().weak(),
                );
            }
        })
        .response;
    ui.add_space(6.0);
    response.interact(Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand)
}
