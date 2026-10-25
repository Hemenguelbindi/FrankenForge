//! Hardware Preflight panel: what the machine reports, a model's components, and an estimate of
//! what loading or training it would need. Hardware is read in the background once and then only
//! on request, never per frame; the estimate is pure arithmetic over cached facts.

mod report_view;
pub(crate) mod state;

use std::fs;
use std::path::PathBuf;

use eframe::egui::{self, RichText};
use ff_core::ModelFormat;
use ff_io::Safetensors;
use ff_preflight::attention::Attention;
use ff_preflight::request::components;
use ff_preflight::{HardwareSnapshot, Known, Placement, Precision, probe};

use crate::failure::Failure;
use crate::forge::palette::Palette;
use crate::forge::widgets::{card, failure_box, muted, pair, title};
use crate::i18n::I18n;
use crate::loader::{Loader, Outcome};
use state::{Opened, Scenario, Settings};

#[derive(Default)]
pub struct PreflightUi {
    probe: Loader<u64, HardwareSnapshot>,
    probes: u64,
    snapshot: Option<HardwareSnapshot>,
    open: Loader<PathBuf, Opened>,
    model: Option<Outcome<Opened>>,
    settings: Settings,
}

impl PreflightUi {
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some((_, Ok(s))) = self.probe.poll() {
            self.snapshot = Some(s);
        }
        if let Some((_, outcome)) = self.open.poll() {
            self.settings.excluded.clear();
            self.model = Some(outcome);
        }
        // Read the hardware once when the panel is first shown; afterwards only on request.
        if self.probes == 0 {
            self.reprobe(ctx);
        }
    }

    fn reprobe(&mut self, ctx: &egui::Context) {
        self.probes += 1;
        let ctx = ctx.clone();
        self.probe.start(self.probes, |_| Ok(probe::probe()), move || ctx.request_repaint());
    }
}

/// Headers and config.json only; no weights are read.
fn open_model(path: &PathBuf) -> Outcome<Opened> {
    let src = Safetensors.open(path).map_err(|e| Failure::from_error(&e))?;
    let config = fs::read(path.join("config.json")).map_err(|e| e.to_string());
    let attention = match config.and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
        Ok(v) => Attention::from_config(&v),
        Err(e) => Known::unknown(format!("config.json: {e}")),
    };
    let companion_bytes = fs::read_dir(path).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_none_or(|x| x != "safetensors"))
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    });
    Ok(Opened { path: path.clone(), components: components(src.model()), attention, companion_bytes })
}

pub fn show(ui: &mut egui::Ui, t: &I18n, p: &mut PreflightUi) {
    let pal = Palette::of(ui.visuals());
    let frame = egui::Frame::central_panel(ui.style()).fill(pal.surface).inner_margin(18);
    egui::CentralPanel::default().frame(frame).show(ui, |ui| {
        p.poll(ui.ctx());
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            ui.label(RichText::new(t.t("pf-title")).size(24.0).strong().color(pal.text));
            muted(ui, pal, t.t("pf-intro"));
            ui.add_space(10.0);
            pair(ui, 340.0, p, |ui, p| hardware(ui, t, pal, p), |ui, p| model(ui, t, pal, p));
            ui.add_space(12.0);
            settings(ui, t, pal, &mut p.settings);
            ui.add_space(12.0);
            report_view::show(ui, t, pal, p);
        });
    });
}

fn known_text(t: &I18n, k: Known<String>) -> String {
    match k {
        Known::Known(v) => v,
        Known::Unknown(why) => t.ta("pf-unknown-why", &[("why", why.into())]),
    }
}

fn free_of(t: &I18n, free: u64, total: u64) -> String {
    t.ta("pf-free-of", &[("free", t.bytes(free).into()), ("total", t.bytes(total).into())])
}

fn fact(ui: &mut egui::Ui, pal: Palette, label: String, value: String) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).color(pal.muted));
        ui.label(RichText::new(value).color(pal.text));
    });
}

fn hardware(ui: &mut egui::Ui, t: &I18n, pal: Palette, p: &mut PreflightUi) {
    card(ui, pal, Some(pal.base), |ui| {
        title(ui, pal, t.t("pf-hw-title"));
        if p.probe.pending().is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(t.t("pf-hw-reading")).color(pal.text));
            });
        }
        if let Some(s) = &p.snapshot {
            let cores = s.logical_cores.get().map_or(String::new(), |n| format!(" · {n}"));
            fact(ui, pal, t.t("pf-hw-cpu"), known_text(t, s.cpu.clone()) + &cores);
            let ram = match (&s.ram_available, &s.ram_total) {
                (Known::Known(free), Known::Known(total)) => free_of(t, *free, *total),
                (Known::Unknown(why), _) | (_, Known::Unknown(why)) => {
                    known_text(t, Known::Unknown(why.clone()))
                }
            };
            fact(ui, pal, t.t("pf-hw-ram"), ram);
            let gpu = s.gpu().map(|g| format!("{} · {}", g.name, free_of(t, g.vram_free, g.vram_total)));
            fact(ui, pal, t.t("pf-hw-gpu"), known_text(t, gpu));
            match &s.disks {
                Known::Known(disks) => {
                    for d in disks {
                        fact(ui, pal, d.mount.display().to_string(), free_of(t, d.available, d.total));
                    }
                }
                Known::Unknown(why) => {
                    fact(ui, pal, t.t("pf-hw-disks"), known_text(t, Known::Unknown(why.clone())))
                }
            }
            let secs = s.taken_at_unix % 86_400;
            let time = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
            muted(ui, pal, t.ta("pf-hw-taken", &[("time", time.into())]));
            let sources = format!("{} · {} · {}", s.ram_source, s.gpu_source, s.disk_source);
            ui.label(RichText::new(sources).monospace().small().color(pal.muted));
        }
        if ui.add_enabled(p.probe.pending().is_none(), egui::Button::new(t.t("pf-hw-reprobe"))).clicked() {
            p.reprobe(ui.ctx());
        }
    });
}

fn model(ui: &mut egui::Ui, t: &I18n, pal: Palette, p: &mut PreflightUi) {
    card(ui, pal, Some(pal.donor), |ui| {
        title(ui, pal, t.t("pf-model-title"));
        muted(ui, pal, t.t("pf-model-hint"));
        if ui.button(t.t("forge-choose-folder")).clicked()
            && let Some(path) = rfd::FileDialog::new().set_title(t.t("pf-model-dialog")).pick_folder()
        {
            let ctx = ui.ctx().clone();
            p.open.start(path, open_model, move || ctx.request_repaint());
        }
        if let Some(path) = p.open.pending() {
            ui.horizontal_wrapped(|ui| {
                ui.spinner();
                ui.label(RichText::new(path.display().to_string()).monospace().color(pal.text));
            });
            return;
        }
        match &p.model {
            None => {}
            Some(Err(failure)) => failure_box(ui, t, pal, t.t("pf-model-failed"), failure, "pf-model"),
            Some(Ok(m)) => {
                ui.label(RichText::new(m.path.display().to_string()).monospace().color(pal.text));
                if let Known::Unknown(why) = &m.attention {
                    muted(ui, pal, t.ta("pf-model-shape-unknown", &[("why", why.clone().into())]));
                }
                muted(ui, pal, t.t("pf-components"));
                for c in &m.components {
                    let mut on = !p.settings.excluded.contains(&c.name);
                    let args = [
                        ("name", c.name.clone().into()),
                        ("params", t.count(c.params as usize).into()),
                        ("size", t.bytes(c.stored_bytes).into()),
                    ];
                    if ui.checkbox(&mut on, t.ta("pf-component", &args)).changed() {
                        if on {
                            p.settings.excluded.remove(&c.name);
                        } else {
                            p.settings.excluded.insert(c.name.clone());
                        }
                    }
                }
            }
        }
    });
}

fn settings(ui: &mut egui::Ui, t: &I18n, pal: Palette, s: &mut Settings) {
    card(ui, pal, None, |ui| {
        title(ui, pal, t.t("pf-settings-title"));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut s.scenario, Scenario::Inference, t.t("pf-scenario-inference"));
            ui.selectable_value(&mut s.scenario, Scenario::Training, t.t("pf-scenario-training"));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(t.t("pf-placement")).color(pal.muted));
            ui.selectable_value(&mut s.placement, Placement::Gpu, t.t("pf-placement-gpu"));
            ui.selectable_value(&mut s.placement, Placement::CpuOffload, t.t("pf-placement-cpu"));
        });
        if s.scenario == Scenario::Inference {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(t.t("pf-precision")).color(pal.muted));
                ui.selectable_value(&mut s.precision, Precision::Bf16, "BF16");
                ui.selectable_value(&mut s.precision, Precision::F32, "F32");
                ui.selectable_value(&mut s.precision, Precision::Int8, t.t("pf-precision-int8"));
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(t.t("pf-context")).color(pal.muted));
            ui.add(egui::DragValue::new(&mut s.context).range(1..=1_048_576).speed(16));
            ui.label(RichText::new(t.t("pf-batch")).color(pal.muted));
            ui.add(egui::DragValue::new(&mut s.batch).range(1..=1024));
        });
        if s.scenario == Scenario::Training {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(t.t("pf-checkpoints")).color(pal.muted));
                ui.add(egui::DragValue::new(&mut s.checkpoints).range(1..=100));
                if ui.button(t.t("pf-checkpoint-dir")).clicked()
                    && let Some(dir) =
                        rfd::FileDialog::new().set_title(t.t("pf-checkpoint-dir")).pick_folder()
                {
                    s.checkpoint_dir = Some(dir);
                }
                let dir = s
                    .checkpoint_dir
                    .as_ref()
                    .map_or_else(|| t.t("forge-not-chosen"), |d| d.display().to_string());
                ui.label(RichText::new(dir).monospace().color(pal.text));
            });
            muted(ui, pal, t.t("pf-training-note"));
        }
    });
}

#[cfg(test)]
#[path = "tests/preflight.rs"]
mod tests;
