//! "What is inside" the selected node, in words: a plain explanation of the part, its share of
//! the model, and for word tables and blocks a question the weights can answer (similar words,
//! words the strongest neurons push toward). Answers come from `ff_surgery::lens` on a worker
//! thread; nothing here reads weights on the UI thread.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use eframe::egui::{self, RichText};
use ff_io::Safetensors;
use ff_surgery::lens::{self, Hit, Lens, NeuronWords, Watch};

use super::canvas::{GraphUi, name};
use super::net::What;
use crate::failure::{Failure, FailureKind};
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::loader::{Loader, Outcome};
use crate::semantics::{Part, Role, classify};
use crate::structure::{group_keys, role_keys};

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// Tokens near `word` in `table`, over one block's slice of columns when `slice` is set.
    Words {
        model: PathBuf,
        table: String,
        word: String,
        slice: Option<(usize, usize)>,
    },
    Neurons {
        model: PathBuf,
        down: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Words(Vec<Hit>),
    /// The word is several tokens in this model; these tokens start it.
    NotOneToken(Vec<String>),
    Neurons(Vec<NeuronWords>),
}

#[derive(Default, Clone)]
struct Meter {
    done: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct InsideState {
    word: String,
    slice_block: usize,
    loader: Loader<Query, Answer>,
    meter: Meter,
    answer: Option<(Query, Outcome<Answer>)>,
}

impl InsideState {
    fn start(&mut self, q: Query, ctx: &egui::Context) {
        self.meter.stop.store(true, Ordering::Relaxed);
        self.meter = Meter::default();
        let (meter, ctx) = (self.meter.clone(), ctx.clone());
        self.loader.start(q, move |q| answer(q, &meter), move || ctx.request_repaint());
    }
}

fn answer(q: &Query, m: &Meter) -> Outcome<Answer> {
    let fail = |e: ff_core::Error| Failure::from_error(&e);
    let watch = Watch { progress: &m.done, stop: &m.stop };
    let stopped = || Failure { kind: FailureKind::Cancelled, detail: "stopped".into() };
    // shortcut: the tokenizer is parsed again per question (about a second for big vocabularies);
    // keep an opened Lens in the state if that ever feels slow.
    match q {
        Query::Words { model, table, word, slice } => {
            let lens = Lens::open(model, &Safetensors).map_err(fail)?;
            let Some(id) = lens.vocab.find(word) else {
                let pieces = lens.vocab.pieces_of(word, 8).into_iter().map(|(_, t)| t).collect();
                return Ok(Answer::NotOneToken(pieces));
            };
            let width = lens.source.model().tensor(table).map_err(fail)?.shape[1];
            let cols = match slice {
                Some((block, blocks)) if width % blocks == 0 => {
                    block * (width / blocks)..(block + 1) * (width / blocks)
                }
                _ => 0..width,
            };
            m.total.store(lens::nearest_bytes(lens.source.model(), table), Ordering::Relaxed);
            let hits = lens::nearest(&*lens.source, table, &lens.vocab, id, cols, 12, watch).map_err(fail)?;
            hits.map(Answer::Words).ok_or_else(stopped)
        }
        Query::Neurons { model, down } => {
            let lens = Lens::open(model, &Safetensors).map_err(fail)?;
            m.total.store(lens::neurons_bytes(lens.source.model(), down, &lens.table), Ordering::Relaxed);
            let found =
                lens::neurons(&*lens.source, down, &lens.table, &lens.vocab, 16, 6, watch).map_err(fail)?;
            found.map(Answer::Neurons).ok_or_else(stopped)
        }
    }
}

pub fn show(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    g: &mut GraphUi,
    picked: &mut Option<String>,
    open_technical: &mut bool,
) {
    if let Some((q, outcome)) = g.inside.loader.poll() {
        g.inside.answer = Some((q, outcome));
    }
    summary(ui, t, inspected, g);
    ui.separator();
    let Some(i) = g.selected.filter(|&i| i < g.net.nodes.len()) else {
        ui.label(RichText::new(t.t("inside-pick")).size(15.0));
        return;
    };
    let node = g.net.nodes[i].clone();
    ui.label(RichText::new(name(t, &node.what)).size(20.0).strong());
    let share = node.params as f64 * 100.0 / inspected.params.max(1) as f64;
    let size = t.ta(
        "inside-size",
        &[("params", t.count(node.params).into()), ("share", format!("{share:.1}").into())],
    );
    ui.label(RichText::new(size).size(15.0));
    ui.add_space(6.0);
    let text = |ui: &mut egui::Ui, s: String| ui.label(RichText::new(s).size(15.0));
    let model = inspected.path.clone();
    let mut ask = None;
    match &node.what {
        What::Words(n) => {
            let key = if n.role == Role::OutputHead { "inside-output-words" } else { "inside-words" };
            text(ui, t.ta(key, &[("rows", n.shape.first().copied().unwrap_or(0).into())]));
            tensor_name(ui, &n.name);
            ask = word_box(ui, t, &mut g.inside, None).map(|word| Query::Words {
                model,
                table: n.name.clone(),
                word,
                slice: None,
            });
        }
        What::PerLayer { node: n, blocks } => {
            text(ui, t.ta("inside-per-layer", &[("blocks", (*blocks).into())]));
            tensor_name(ui, &n.name);
            if n.shape.len() == 2 && *blocks > 0 {
                let blocks = *blocks;
                ask = word_box(ui, t, &mut g.inside, Some(blocks)).map(|word| Query::Words {
                    model,
                    table: n.name.clone(),
                    word,
                    slice: Some((g.inside.slice_block.min(blocks - 1), blocks)),
                });
            }
        }
        What::Block { layout, stack, index, shared_memory, kind } => {
            text(ui, t.ta("inside-block", &[("stack", stack.clone().into()), ("index", (*index).into())]));
            if let Some(k) = kind {
                text(ui, t.ta("inside-block-kind", &[("kind", k.clone().into())]));
            }
            if *shared_memory {
                ui.label(
                    RichText::new(t.t("inside-shared-memory")).size(15.0).color(ui.visuals().warn_fg_color),
                );
            }
            parts(ui, t, inspected, *layout, picked, open_technical);
            let down = inspected.layout.blocks[*layout]
                .tensors
                .iter()
                .find(|n| classify(n).role == Role::Mlp(Part::Down))
                .cloned();
            ui.add_space(6.0);
            match down {
                Some(down) => {
                    text(ui, t.t("inside-neurons-hint"));
                    if ui.button(RichText::new(t.t("inside-neurons-button")).size(15.0).strong()).clicked() {
                        ask = Some(Query::Neurons { model, down });
                    }
                }
                None => {
                    text(ui, t.t("inside-neurons-none"));
                }
            }
        }
        What::Bridge(n) => {
            text(ui, t.t("inside-bridge"));
            tensor_name(ui, &n.name);
        }
        What::Output(n) | What::Input(n) => {
            text(ui, t.t(role_keys(n.role).1));
            tensor_name(ui, &n.name);
        }
    }
    if let Some(q) = ask {
        g.inside.start(q, ui.ctx());
    }
    ui.add_space(8.0);
    results(ui, t, &mut g.inside);
}

/// The landmarks the program found, each a button that selects its node.
fn summary(ui: &mut egui::Ui, t: &I18n, inspected: &Inspected, g: &mut GraphUi) {
    ui.label(RichText::new(t.t("inside-summary")).size(16.0).strong());
    let Some(m) = g.marks.clone() else { return };
    for (key, node) in [
        ("inside-largest", m.largest),
        ("inside-entry", m.entry),
        ("inside-exit", m.exit),
        ("inside-most-linked", m.most_linked),
    ] {
        let Some(i) = node else { continue };
        let n = &g.net.nodes[i];
        let share = format!("{:.0}", n.params as f64 * 100.0 / inspected.params.max(1) as f64);
        let label = t.ta(key, &[("name", name(t, &n.what).into()), ("share", share.into())]);
        if ui.link(RichText::new(label).size(15.0)).clicked() {
            g.selected = Some(i);
        }
    }
}

fn tensor_name(ui: &mut egui::Ui, name: &str) {
    ui.add(egui::Label::new(RichText::new(name).monospace().weak()).selectable(true));
}

/// A word field and the button; returns the word when asked. `blocks` adds a block chooser.
fn word_box(ui: &mut egui::Ui, t: &I18n, s: &mut InsideState, blocks: Option<usize>) -> Option<String> {
    ui.add_space(6.0);
    ui.label(RichText::new(t.t("inside-word-label")).size(15.0));
    let edit = ui.add(egui::TextEdit::singleline(&mut s.word).font(egui::FontId::proportional(16.0)));
    if let Some(n) = blocks {
        ui.add(egui::Slider::new(&mut s.slice_block, 0..=n.saturating_sub(1)).text(t.t("inside-for-block")));
    }
    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    let clicked = ui.button(RichText::new(t.t("inside-words-button")).size(15.0).strong()).clicked();
    ((clicked || enter) && !s.word.trim().is_empty()).then(|| s.word.trim().to_owned())
}

/// The block's parts by purpose, with sizes.
/// Clicking a part opens that tensor in Technical data.
fn parts(
    ui: &mut egui::Ui,
    t: &I18n,
    inspected: &Inspected,
    layout: usize,
    picked: &mut Option<String>,
    open_technical: &mut bool,
) {
    let towers = super::towers(inspected);
    let width =
        towers.iter().find(|w| w.stacks.iter().any(|s| s.blocks.contains(&layout))).and_then(|w| w.width);
    let g = super::block(inspected, layout, width);
    for b in &g.branches {
        ui.label(RichText::new(t.t(group_keys(b.group).0)).size(15.0).strong());
        for m in &b.matrices {
            let flow = match &m.flow {
                super::Flow::Reads { from, to } | super::Flow::Writes { from, to } => {
                    t.ta("graph-flow", &[("from", (*from).into()), ("to", (*to).into())])
                }
                super::Flow::Shape(s) => s.iter().map(usize::to_string).collect::<Vec<_>>().join(" × "),
            };
            let count = if m.count > 1 { format!("  × {}", m.count) } else { String::new() };
            let line = format!("  {}   {flow}{count}", t.t(role_keys(m.role).0));
            if ui.link(RichText::new(line).size(14.0)).on_hover_text(&m.name).clicked() {
                *picked = Some(m.name.clone());
                *open_technical = true;
            }
        }
    }
    if g.norms > 0 {
        ui.label(RichText::new(t.ta("graph-norms", &[("count", g.norms.into())])).size(14.0).weak());
    }
}

fn results(ui: &mut egui::Ui, t: &I18n, s: &mut InsideState) {
    if s.loader.pending().is_some() {
        let total = s.meter.total.load(Ordering::Relaxed).max(1);
        let fraction = s.meter.done.load(Ordering::Relaxed) as f32 / total as f32;
        ui.add(egui::ProgressBar::new(fraction.min(1.0)).text(t.t("inside-reading")));
        ui.ctx().request_repaint_after(Duration::from_millis(100));
        if ui.button(t.t("inside-stop")).clicked() {
            s.meter.stop.store(true, Ordering::Relaxed);
        }
        return;
    }
    let mut retry = None;
    match &s.answer {
        None => {}
        Some((_, Err(f))) if f.kind == FailureKind::Cancelled => {
            ui.label(t.t("inside-stopped"));
        }
        Some((_, Err(f))) => {
            ui.label(RichText::new(t.t("inside-failed")).color(ui.visuals().error_fg_color));
            ui.label(RichText::new(&f.detail).monospace().small());
        }
        Some((q, Ok(Answer::Words(hits)))) => {
            if let Query::Words { word, .. } = q {
                ui.label(
                    RichText::new(t.ta("inside-near", &[("word", word.clone().into())])).size(16.0).strong(),
                );
            }
            ui.horizontal_wrapped(|ui| {
                for h in hits.iter().skip(1) {
                    ui.label(RichText::new(format!("{}  {:.2}", h.text.trim(), h.score)).size(16.0))
                        .on_hover_text(format!("{:?} #{}", h.text, h.id));
                    ui.add_space(10.0);
                }
            });
            ui.label(RichText::new(t.t("inside-near-note")).size(13.0).weak());
        }
        Some((q, Ok(Answer::NotOneToken(pieces)))) => {
            ui.label(RichText::new(t.t("inside-not-one-token")).size(15.0));
            ui.horizontal_wrapped(|ui| {
                for p in pieces {
                    if ui.button(RichText::new(p.trim()).size(15.0)).clicked()
                        && let Query::Words { model, table, slice, .. } = q
                    {
                        retry = Some((p.trim().to_owned(), model.clone(), table.clone(), *slice));
                    }
                }
            });
        }
        Some((_, Ok(Answer::Neurons(found)))) => {
            ui.label(RichText::new(t.t("inside-neurons-title")).size(16.0).strong());
            for n in found {
                let words: Vec<&str> =
                    n.words.iter().map(|h| h.text.trim()).filter(|w| !w.is_empty()).collect();
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(t.ta("inside-neuron", &[("index", n.neuron.into())])).size(14.0).weak(),
                    );
                    ui.label(RichText::new(words.join(" · ")).size(16.0));
                });
            }
            ui.label(RichText::new(t.t("inside-neurons-note")).size(13.0).weak());
        }
    }
    if let Some((word, model, table, slice)) = retry {
        s.word = word.clone();
        let ctx = ui.ctx().clone();
        s.start(Query::Words { model, table, word, slice }, &ctx);
    }
}
