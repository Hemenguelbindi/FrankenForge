//! Output-location safety: the result must be a new path that is not inside the Base or Donor
//! model, so a build can never modify a source, even partially.

use std::io;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputProblem {
    /// Not a single plain folder name (empty, a separator, `.` or `..`).
    BadName,
    /// This path is already taken; nothing is ever overwritten.
    Exists(PathBuf),
    /// The output would land inside a source model (also through a link or junction).
    InsideSource,
    /// The location cannot be resolved (missing drive, no access).
    Unresolved(String),
    /// Hardware Preflight: the volume has less free space than the result needs.
    NoSpace { need: u64, available: u64 },
}

pub const RECIPE: &str = ".recipe.toml";
pub const LOG: &str = ".build.log";

/// `out` + `suffix`, e.g. `model` -> `model.recipe.toml` next to it.
pub fn sibling(out: &Path, suffix: &str) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

/// Every path a build creates next to `out` (the lock suffix is the core's `Lock::path_for`);
/// each must be free before anything is written.
pub fn artifacts(out: &Path) -> Vec<PathBuf> {
    [".partial", ".lock.json", RECIPE, LOG].iter().map(|s| sibling(out, s)).chain([out.to_owned()]).collect()
}

/// A new folder name: exactly one normal path component.
pub fn valid_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!((parts.next(), parts.next()), (Some(Component::Normal(_)), None)) && name.trim() == name
}

/// Resolves a path that may not exist yet: the deepest existing ancestor is canonicalized (links
/// and junctions followed), the rest is appended with `.` and `..` applied. `canonicalize` alone
/// fails on Windows for a path that does not exist.
pub fn resolve(path: &Path) -> io::Result<PathBuf> {
    let path = std::path::absolute(path)?;
    let mut existing = path.as_path();
    let mut rest = Vec::new();
    while !existing.exists() {
        rest.push(existing.file_name().ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?);
        existing = existing.parent().ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;
    }
    let mut out = existing.canonicalize()?;
    for part in rest.iter().rev() {
        match Path::new(part).components().next() {
            Some(Component::ParentDir) => {
                out.pop();
            }
            Some(Component::Normal(p)) => out.push(p),
            _ => {}
        }
    }
    Ok(out)
}

/// Windows paths compare case-insensitively.
fn key(p: &Path) -> PathBuf {
    if cfg!(windows) { PathBuf::from(p.to_string_lossy().to_lowercase()) } else { p.to_owned() }
}

/// `out` and its artifacts must not exist, and `out` must not be inside (or equal to) a source.
/// Sources exist, so "must not exist" also rules out writing over a source file.
pub fn check_output(out: &Path, sources: &[&Path]) -> Result<(), OutputProblem> {
    if let Some(taken) = artifacts(out).into_iter().find(|p| p.exists() || p.is_symlink()) {
        return Err(OutputProblem::Exists(taken));
    }
    let unresolved = |e: io::Error| OutputProblem::Unresolved(e.to_string());
    let out = key(&resolve(out).map_err(unresolved)?);
    for source in sources {
        let source = key(&resolve(source).map_err(unresolved)?);
        if out.starts_with(&source) || source.starts_with(&out) {
            return Err(OutputProblem::InsideSource);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/paths.rs"]
mod tests;
