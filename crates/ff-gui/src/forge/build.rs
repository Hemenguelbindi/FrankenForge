//! The build job: re-checks the output location on disk, writes the recipe, runs the core's
//! `build::build` (which keeps `.partial`, verify and the lock file) and writes a build log.
//! Nothing here can overwrite a file: every write is `create_new`.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ff_core::{Error, Result};
use ff_io::Safetensors;
use ff_surgery::build::{self, BuildReport};
use ff_surgery::lock::Lock;

use ff_preflight::{Known, PreflightStatus, probe};

use super::paths::{self, OutputProblem};
use super::readiness::disk_status;
use super::state::{Preview, Selection, recipe};
use crate::failure::Failure;
use crate::loader::Outcome;

/// What to build; the preview is the real plan of the same selection, shown again in the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildRequest {
    pub selection: Selection,
    pub out: PathBuf,
    pub preview: Preview,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Built {
    pub out: PathBuf,
    pub recipe: PathBuf,
    pub lock: PathBuf,
    /// `None` if the log could not be written (the reason is in `warnings`).
    pub log: Option<PathBuf>,
    pub tensors: usize,
    pub params: usize,
    /// Tensors per source, from the plan the build used.
    pub planned: Preview,
    pub warnings: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The fresh check before writing found the location unsafe; nothing was written.
    Refused(OutputProblem),
    /// The core failed; the files kept for diagnosis are listed.
    Failed { failure: Failure, recipe: Option<PathBuf>, log: Option<PathBuf> },
}

pub type BuildResult = std::result::Result<Built, BuildError>;

/// Loader entry point; the outer error only reports a crashed worker.
pub fn run(req: &BuildRequest) -> Outcome<BuildResult> {
    Ok(build(req))
}

fn build(req: &BuildRequest) -> BuildResult {
    let pair = &req.selection.pair;
    // The cached readiness check is not enough: the disk may have changed since.
    paths::check_output(&req.out, &[&pair.base, &pair.donor]).map_err(BuildError::Refused)?;
    // A known shortage of free space refuses now; Unknown proceeds and the core still fails closed.
    let available = probe::disk_available(&req.out);
    if let (PreflightStatus::Insufficient, need) = disk_status(req.preview.output_bytes, &req.out, &available)
        && let Known::Known(available) = available
    {
        return Err(BuildError::Refused(OutputProblem::NoSpace { need, available }));
    }

    let failed = |e: &Error, recipe: Option<PathBuf>, log: Option<PathBuf>| BuildError::Failed {
        failure: Failure::from_error(e),
        recipe,
        log,
    };
    let recipe_path = paths::sibling(&req.out, paths::RECIPE);
    recipe(&req.selection)
        .to_toml()
        .and_then(|text| write_new(&recipe_path, &text))
        .map_err(|e| failed(&e, None, None))?;

    let result = build::build(&recipe_path, &req.out, &Safetensors, false);
    let log_path = paths::sibling(&req.out, paths::LOG);
    let log = write_new(&log_path, &log_text(req, &recipe_path, &result));
    match result {
        Ok(report) => {
            let mut warnings = report.warnings;
            if let Err(e) = &log {
                warnings.push(format!("build log not written: {e}"));
            }
            Ok(Built {
                out: req.out.clone(),
                recipe: recipe_path,
                lock: report.lock_path,
                log: log.ok().map(|()| log_path),
                tensors: report.tensors,
                params: report.params,
                planned: req.preview.clone(),
                warnings,
            })
        }
        // The recipe and the log stay next to the output for diagnosis.
        Err(e) => Err(failed(&e, Some(recipe_path), log.ok().map(|()| log_path))),
    }
}

/// Creates `path`; fails if anything is already there, so nothing is ever overwritten.
fn write_new(path: &Path, text: &str) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(Error::io(path))?;
    file.write_all(text.as_bytes()).map_err(Error::io(path))
}

fn log_text(req: &BuildRequest, recipe: &Path, result: &Result<BuildReport>) -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let blocks: Vec<String> = req.selection.blocks.iter().map(|b| b.prefix()).collect();
    let mut s = String::new();
    let _ = writeln!(s, "FrankenForge GUI build log");
    let _ = writeln!(s, "tool version: {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(s, "finished at (unix seconds): {secs}");
    let _ = writeln!(s, "base: {}", req.selection.pair.base.display());
    let _ = writeln!(s, "donor: {}", req.selection.pair.donor.display());
    let _ = writeln!(s, "blocks from donor (same index): {}", blocks.join(", "));
    let _ = writeln!(
        s,
        "planned tensors: {} from base, {} from donor",
        req.preview.from_base, req.preview.from_donor
    );
    let _ = writeln!(s, "recipe: {}", recipe.display());
    let _ = writeln!(s, "output: {}", req.out.display());
    match result {
        Ok(r) => {
            let _ = writeln!(
                s,
                "result: OK, written and verified ({} tensors, {} parameters)",
                r.tensors, r.params
            );
            let _ = writeln!(s, "lock: {}", r.lock_path.display());
            for w in &r.warnings {
                let _ = writeln!(s, "warning: {w}");
            }
            let _ = writeln!(
                s,
                "note: verified means the file matches the plan; model quality was not evaluated"
            );
        }
        Err(e) => {
            let _ = writeln!(s, "result: FAILED");
            let _ = writeln!(s, "error: {e}");
            let leftover = [req.out.clone(), Lock::path_for(&req.out)].into_iter().filter(|p| p.exists());
            for p in leftover {
                let _ = writeln!(s, "left on disk (not a verified model): {}", p.display());
            }
        }
    }
    s
}

#[cfg(test)]
#[path = "tests/build.rs"]
mod tests;
