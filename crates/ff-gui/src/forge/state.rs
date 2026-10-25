//! What the user chose in Forge, which background results still belong to that choice, and the
//! recipe it describes. No egui here: everything is plain data, tested without a window.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ff_surgery::compare::TokenizerMatch;
use ff_surgery::{Recipe, Step};

use super::analysis::{Analysis, BlockId};
use super::build::{BuildError, BuildRequest, BuildResult};
use super::readiness::{self, Checks, Inputs, Problem};
use crate::loader::Outcome;

/// Source aliases in the generated recipe.
pub const BASE: &str = "base";
pub const DONOR: &str = "donor";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    pub base: PathBuf,
    pub donor: PathBuf,
}

/// Everything a preview or a build depends on; a result computed for another selection is stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub pair: Pair,
    pub blocks: Vec<BlockId>,
}

/// Same index to same index: block N of the Donor replaces block N of the Base.
pub fn recipe(sel: &Selection) -> Recipe {
    let steps = sel
        .blocks
        .iter()
        .map(|b| Step::Transplant { from: DONOR.into(), src: b.prefix(), dst: b.prefix() })
        .collect();
    Recipe {
        name: "forge".into(),
        base: BASE.into(),
        sources: BTreeMap::from([
            (BASE.into(), sel.pair.base.clone()),
            (DONOR.into(), sel.pair.donor.clone()),
        ]),
        metadata: BTreeMap::new(),
        steps,
    }
}

/// Result of the real `build::plan` for a selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preview {
    pub from_base: usize,
    pub from_donor: usize,
    /// Planner notes (config.json changes, warnings), shown as the core wrote them.
    pub notes: Vec<String>,
}

#[derive(Debug, Default)]
pub enum BuildStatus {
    #[default]
    Idle,
    /// Every input is locked until the result arrives.
    Running(BuildRequest),
    /// Kept until the user changes an input.
    Done(BuildResult),
}

#[derive(Default)]
pub struct Forge {
    base: Option<PathBuf>,
    donor: Option<PathBuf>,
    analysis: Option<(Pair, Outcome<Analysis>)>,
    selected: BTreeSet<BlockId>,
    preview: Option<(Selection, Outcome<Preview>)>,
    out_dir: Option<PathBuf>,
    out_name: String,
    /// Disk checks and the inputs they were made for.
    checks: Option<(Inputs, Checks)>,
    build: BuildStatus,
}

impl Forge {
    /// False while a build runs: Base, Donor, blocks and output stay as the build saw them.
    /// Any accepted edit clears the previous build result.
    fn editable(&mut self) -> bool {
        if self.running() {
            return false;
        }
        self.build = BuildStatus::Idle;
        true
    }

    pub fn running(&self) -> bool {
        matches!(self.build, BuildStatus::Running(_))
    }

    pub fn build_status(&self) -> &BuildStatus {
        &self.build
    }

    /// Starts a build if nothing blocks it; the request goes to the background job.
    pub fn begin_build(&mut self) -> Option<BuildRequest> {
        self.begin_build_with(readiness::check)
    }

    pub fn begin_build_with(&mut self, check: impl FnOnce(&Inputs) -> Checks) -> Option<BuildRequest> {
        if self.running() || !self.problems_with(check).is_empty() {
            return None;
        }
        let preview = self.preview()?.as_ref().ok()?.clone();
        let req = BuildRequest { selection: self.selection()?, out: self.out()?, preview };
        self.build = BuildStatus::Running(req.clone());
        Some(req)
    }

    /// A crashed worker counts as a failed build. The disk changed, so the checks run again.
    pub fn finish_build(&mut self, req: BuildRequest, outcome: Outcome<BuildResult>) {
        if matches!(&self.build, BuildStatus::Running(r) if *r == req) {
            let result = outcome
                .unwrap_or_else(|failure| Err(BuildError::Failed { failure, recipe: None, log: None }));
            self.build = BuildStatus::Done(result);
            self.checks = None;
        }
    }

    pub fn out_dir(&self) -> Option<&PathBuf> {
        self.out_dir.as_ref()
    }

    pub fn out_name(&self) -> &str {
        &self.out_name
    }

    pub fn set_out_dir(&mut self, dir: PathBuf) {
        if self.editable() {
            self.out_dir = Some(dir);
        }
    }

    pub fn set_out_name(&mut self, name: String) {
        if self.editable() {
            self.out_name = name;
        }
    }

    pub fn base(&self) -> Option<&PathBuf> {
        self.base.as_ref()
    }

    pub fn donor(&self) -> Option<&PathBuf> {
        self.donor.as_ref()
    }

    /// A new model invalidates the analysis, the chosen blocks and the preview.
    pub fn set_base(&mut self, path: PathBuf) {
        if self.editable() {
            self.base = Some(path);
            self.reset_analysis();
        }
    }

    pub fn set_donor(&mut self, path: PathBuf) {
        if self.editable() {
            self.donor = Some(path);
            self.reset_analysis();
        }
    }

    fn reset_analysis(&mut self) {
        self.analysis = None;
        self.selected.clear();
        self.preview = None;
    }

    pub fn pair(&self) -> Option<Pair> {
        Some(Pair { base: self.base.clone()?, donor: self.donor.clone()? })
    }

    /// A result for a pair that is no longer chosen is dropped.
    pub fn apply_analysis(&mut self, pair: Pair, outcome: Outcome<Analysis>) {
        if self.pair().as_ref() == Some(&pair) {
            self.reset_analysis();
            self.analysis = Some((pair, outcome));
        }
    }

    pub fn analysis(&self) -> Option<&Outcome<Analysis>> {
        self.analysis.as_ref().map(|(_, a)| a)
    }

    #[cfg(test)]
    pub fn is_selected(&self, id: &BlockId) -> bool {
        self.selected.contains(id)
    }

    /// Only structurally compatible blocks can be selected.
    pub fn toggle(&mut self, id: &BlockId) {
        let compatible = matches!(self.analysis(), Some(Ok(a)) if a.compatible(id));
        if self.editable() && !self.selected.remove(id) && compatible {
            self.selected.insert(id.clone());
        }
    }

    pub fn clear_selection(&mut self) {
        if self.editable() {
            self.selected.clear();
        }
    }

    pub fn selection(&self) -> Option<Selection> {
        if self.selected.is_empty() || !matches!(self.analysis(), Some(Ok(_))) {
            return None;
        }
        Some(Selection { pair: self.pair()?, blocks: self.selected.iter().cloned().collect() })
    }

    pub fn apply_preview(&mut self, sel: Selection, outcome: Outcome<Preview>) {
        if self.selection().as_ref() == Some(&sel) {
            self.preview = Some((sel, outcome));
        }
    }

    /// The preview of the current selection; a preview of an older selection is not shown.
    pub fn preview(&self) -> Option<&Outcome<Preview>> {
        let current = self.selection()?;
        self.preview.as_ref().filter(|(sel, _)| *sel == current).map(|(_, p)| p)
    }

    /// The selection whose preview should start now, unless it is already running.
    pub fn preview_needed(&self, pending: Option<&Selection>) -> Option<Selection> {
        let sel = self.selection()?;
        (self.preview().is_none() && pending != Some(&sel)).then_some(sel)
    }

    pub fn out(&self) -> Option<PathBuf> {
        Some(self.out_dir.as_ref()?.join(&self.out_name))
    }

    fn inputs(&self) -> Inputs {
        Inputs {
            base: self.base.clone(),
            donor: self.donor.clone(),
            out_dir: self.out_dir.clone(),
            out_name: self.out_name.clone(),
        }
    }

    /// Everything that blocks a build, in the order the user should fix it.
    pub fn problems(&mut self) -> Vec<Problem> {
        self.problems_with(readiness::check)
    }

    /// `check` runs only when its inputs changed since the last call, so a frame that changes
    /// nothing does no disk I/O.
    pub fn problems_with(&mut self, check: impl FnOnce(&Inputs) -> Checks) -> Vec<Problem> {
        let inputs = self.inputs();
        if self.checks.as_ref().is_none_or(|(i, _)| *i != inputs) {
            let checks = check(&inputs);
            self.checks = Some((inputs, checks));
        }
        let checks = &self.checks.as_ref().expect("filled just above").1;

        let mut out = Vec::new();
        match (&self.base, &self.donor) {
            (None, _) => out.push(Problem::NeedBase),
            (_, None) => out.push(Problem::NeedDonor),
            _ if checks.same_model => out.push(Problem::SameModel),
            _ => {}
        }
        if self.base.is_some() && self.donor.is_some() {
            match self.analysis() {
                None => out.push(Problem::NeedAnalysis),
                Some(Err(_)) => out.push(Problem::AnalysisFailed),
                Some(Ok(a)) if a.report.tokenizer.status != TokenizerMatch::Identical => {
                    out.push(Problem::Tokenizer)
                }
                Some(Ok(_)) if self.selected.is_empty() => out.push(Problem::NeedBlocks),
                Some(Ok(_)) => match self.preview() {
                    None => out.push(Problem::NeedPreview),
                    Some(Err(_)) => out.push(Problem::PreviewFailed),
                    Some(Ok(_)) => {}
                },
            }
        }
        match &checks.output {
            None => out.push(Problem::NeedOutput),
            Some(Err(p)) => out.push(Problem::Output(p.clone())),
            Some(Ok(())) => {}
        }
        out
    }
}

#[cfg(test)]
#[path = "tests/state.rs"]
mod tests;
