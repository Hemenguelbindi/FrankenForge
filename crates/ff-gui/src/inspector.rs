//! Tensor Inspector state: the selected tensor, its single analysis, the display mode, and the
//! deterministic text rule for the numeric profile. Drawing lives in `panel`.

mod panel;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use eframe::egui;
use ff_core::TensorInfo;

use crate::analysis::{self, Analysis, Profile, byte_len};
use crate::components::Kind;
use crate::i18n::I18n;
use crate::inspected::Inspected;
use crate::loader::{Loader, Outcome};
use crate::semantics::{TensorDoc, describe};

pub use panel::panel;

/// How much of the inspector is shown at once. Stored with the other UI settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    #[default]
    Simple,
    Expert,
}

impl Mode {
    pub fn code(self) -> &'static str {
        match self {
            Mode::Simple => "simple",
            Mode::Expert => "expert",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        [Mode::Simple, Mode::Expert].into_iter().find(|m| m.code().eq_ignore_ascii_case(code.trim()))
    }

    pub(crate) fn message_key(self) -> &'static str {
        match self {
            Mode::Simple => "mode-simple",
            Mode::Expert => "mode-expert",
        }
    }
}

/// Everything about the selected tensor that does not depend on the interface language,
/// computed once when the selection changes instead of on every frame.
struct Selection {
    name: String,
    info: TensorInfo,
    doc: TensorDoc,
    component: Option<Kind>,
    /// (stack, block index) when the tensor is inside a repeated block.
    block: Option<(String, usize)>,
}

/// The selected tensor and the single analysis that may be running or shown for it.
#[derive(Default)]
pub struct InspectorState {
    mode: Mode,
    selection: Option<Selection>,
    loader: Loader<String, Analysis>,
    result: Option<(String, Outcome<Analysis>)>,
    progress: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    /// Bytes both passes will read, for the progress bar.
    total: u64,
}

impl InspectorState {
    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    /// Drops the selection and any analysis, running or finished. The worker stops at its next
    /// chunk, and its result could not be delivered anyway. The chosen mode is kept.
    pub fn reset(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.loader.clear();
        self.result = None;
        self.selection = None;
    }

    pub fn selected(&self) -> Option<&str> {
        self.selection.as_ref().map(|s| s.name.as_str())
    }

    /// Selecting another tensor discards the previous one's analysis.
    pub fn select(&mut self, inspected: &Inspected, name: Option<String>) {
        if name.as_deref() == self.selected() {
            return;
        }
        self.reset();
        let Some(name) = name else { return };
        let Some(info) = inspected.model.tensors.get(&name) else { return };
        let block = ff_core::layout::block_of(&name);
        let component = inspected
            .components
            .iter()
            .find(|c| match &block {
                Some((stack, ..)) => c.stacks.iter().any(|s| &s.name == stack),
                None => c.loose.contains(&name),
            })
            .map(|c| c.kind);
        self.selection = Some(Selection {
            doc: describe(info, inspected.config_ref()),
            info: info.clone(),
            block: block.map(|(stack, index, _)| (stack, index)),
            component,
            name,
        });
    }

    fn start(&mut self, ctx: &egui::Context, inspected: &Inspected) {
        let Some(info) = self.selection.as_ref().map(|s| s.info.clone()) else { return };
        self.stop.store(true, Ordering::Relaxed);
        self.result = None;
        self.progress = Arc::new(AtomicU64::new(0));
        self.stop = Arc::new(AtomicBool::new(false));
        self.total = byte_len(&info).map_or(0, |n| n.saturating_mul(2));
        let (progress, stop, ctx) = (self.progress.clone(), self.stop.clone(), ctx.clone());
        let source = inspected.source.clone();
        self.loader.start(
            info.name.clone(),
            move |_| analysis::analyze(&*source, &info, &progress, &stop),
            move || ctx.request_repaint(),
        );
    }

    pub fn poll(&mut self) {
        if let Some(done) = self.loader.poll() {
            self.accept(done);
        }
    }

    /// Abandons a running analysis when the user leaves the Technical view: the worker halts at
    /// its next chunk, and the selection and a finished result survive.
    pub fn stop_analysis(&mut self) {
        if self.loader.pending().is_some() {
            self.stop.store(true, Ordering::Relaxed);
            self.loader.clear();
        }
    }

    /// Stores a finished analysis only for the tensor selected now: a result that arrives after
    /// the selection moved on is dropped instead of being shown for another tensor.
    fn accept(&mut self, done: (String, Outcome<Analysis>)) {
        if self.selected() == Some(done.0.as_str()) {
            self.result = Some(done);
        }
    }
}

/// Deterministic text rules, in a fixed order: non-finite values, then all zeros, then an
/// ordinary profile. `None` only for a profile with no elements, which `analyze` reports as
/// `Empty` instead.
fn conclusion(t: &I18n, p: &Profile) -> Option<String> {
    if p.nan + p.pos_inf + p.neg_inf > 0 {
        return Some(t.ta(
            "conclusion-nonfinite",
            &[("nan", p.nan.into()), ("pos_inf", p.pos_inf.into()), ("neg_inf", p.neg_inf.into())],
        ));
    }
    if p.count > 0 && p.zeros == p.count {
        return Some(t.ta("conclusion-all-zero", &[("count", p.count.into())]));
    }
    let f = p.finite.as_ref()?;
    Some(t.ta(
        "conclusion-profile",
        &[
            ("min", number(t, f.min).into()),
            ("max", number(t, f.max).into()),
            ("mean", number(t, f.mean).into()),
            ("std", number(t, f.std).into()),
        ],
    ))
}

/// Profile numbers use the locale's decimal separator, like sizes elsewhere in the UI.
fn number(t: &I18n, v: f64) -> String {
    let text = if v == 0.0 {
        "0".into()
    } else if (1e-3..1e5).contains(&v.abs()) {
        format!("{v:.6}")
    } else {
        format!("{v:.4e}")
    };
    text.replace('.', &t.t("number-decimal-separator"))
}

fn percent(t: &I18n, part: u64, whole: u64) -> String {
    let text = if whole == 0 { "0".into() } else { format!("{:.2}", part as f64 * 100.0 / whole as f64) };
    text.replace('.', &t.t("number-decimal-separator"))
}

#[cfg(test)]
#[path = "tests/inspector.rs"]
mod tests;
