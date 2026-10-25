//! Why a build cannot start yet. The checks that touch the disk (same model twice, a safe output
//! location) run only when their inputs change; see `Forge::problems_with`.

use std::path::{Path, PathBuf};

use super::paths::{self, OutputProblem};

/// Everything the disk checks depend on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub base: Option<PathBuf>,
    pub donor: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
    pub out_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checks {
    pub same_model: bool,
    /// `None` until an output location is chosen.
    pub output: Option<Result<(), OutputProblem>>,
}

/// Reads the file system: resolves both models and checks that the output is new and outside them.
pub fn check(i: &Inputs) -> Checks {
    let same_model = match (&i.base, &i.donor) {
        (Some(b), Some(d)) => match (paths::resolve(b), paths::resolve(d)) {
            (Ok(b), Ok(d)) => b == d,
            _ => b == d,
        },
        _ => false,
    };
    let output = i.out_dir.as_ref().map(|dir| {
        if !paths::valid_name(&i.out_name) {
            return Err(OutputProblem::BadName);
        }
        let sources: Vec<&Path> = i.base.iter().chain(&i.donor).map(PathBuf::as_path).collect();
        paths::check_output(&dir.join(&i.out_name), &sources)
    });
    Checks { same_model, output }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    NeedBase,
    NeedDonor,
    SameModel,
    NeedAnalysis,
    AnalysisFailed,
    Tokenizer,
    NeedBlocks,
    NeedPreview,
    PreviewFailed,
    NeedOutput,
    Output(OutputProblem),
}

impl Problem {
    #[cfg(test)]
    pub const KEYS: [&str; 14] = [
        "forge-need-base",
        "forge-need-donor",
        "forge-same-model",
        "forge-need-analysis",
        "forge-analysis-error",
        "forge-tokenizer-blocked",
        "forge-need-blocks",
        "forge-need-preview",
        "forge-preview-error",
        "forge-need-output",
        "forge-bad-name",
        "forge-output-exists",
        "forge-output-inside-source",
        "forge-output-unresolved",
    ];

    pub fn message_key(&self) -> &'static str {
        match self {
            Problem::NeedBase => "forge-need-base",
            Problem::NeedDonor => "forge-need-donor",
            Problem::SameModel => "forge-same-model",
            Problem::NeedAnalysis => "forge-need-analysis",
            Problem::AnalysisFailed => "forge-analysis-error",
            Problem::Tokenizer => "forge-tokenizer-blocked",
            Problem::NeedBlocks => "forge-need-blocks",
            Problem::NeedPreview => "forge-need-preview",
            Problem::PreviewFailed => "forge-preview-error",
            Problem::NeedOutput => "forge-need-output",
            Problem::Output(OutputProblem::BadName) => "forge-bad-name",
            Problem::Output(OutputProblem::Exists(_)) => "forge-output-exists",
            Problem::Output(OutputProblem::InsideSource) => "forge-output-inside-source",
            Problem::Output(OutputProblem::Unresolved(_)) => "forge-output-unresolved",
        }
    }
}

/// The Build button is clickable only with nothing left to fix and no build running.
pub fn build_enabled(problems: &[Problem], running: bool) -> bool {
    problems.is_empty() && !running
}

#[cfg(test)]
#[path = "tests/readiness.rs"]
mod tests;
