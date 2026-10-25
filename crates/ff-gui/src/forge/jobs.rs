//! Work that runs on a `Loader` thread. Each job only calls an ff-surgery use case and converts
//! its result for the view; no comparison or planning logic lives here.

use std::fs;
use std::path::Path;

use ff_core::Error;
use ff_io::Safetensors;
use ff_surgery::{build, compare};

use super::analysis::Analysis;
use super::state::{BASE, DONOR, Pair, Preview, Selection, recipe};
use crate::failure::Failure;
use crate::loader::Outcome;

/// Structural comparison of the two models: headers and tokenizer.json only, no weights.
pub fn analyze(pair: &Pair) -> Outcome<Analysis> {
    compare::run(&pair.base, &pair.donor, &Safetensors)
        .map(Analysis::new)
        .map_err(|e| Failure::from_error(&e))
}

/// The real `build::plan` for the selection: metadata only, nothing is written.
pub fn preview(sel: &Selection) -> Outcome<Preview> {
    // Source paths in the recipe are absolute, so they need no base directory.
    let (_, plan) =
        build::plan(&recipe(sel), Path::new(""), &Safetensors).map_err(|e| Failure::from_error(&e))?;
    let provenance = plan.provenance();
    let mut notes = plan.companion.changes.clone();
    notes.extend(build::warnings(&plan));
    let weights: u64 = plan.tensors.values().map(|p| p.info.byte_len() as u64).sum();
    // Companions are copied byte for byte from the base directory.
    let mut companions = 0;
    if let Some(dir) = &plan.companion.dir {
        for name in &plan.companion.files {
            let path = dir.join(name);
            companions += fs::metadata(&path).map_err(|e| Failure::from_error(&Error::io(&path)(e)))?.len();
        }
    }
    Ok(Preview {
        from_base: provenance.get(BASE).copied().unwrap_or(0),
        from_donor: provenance.get(DONOR).copied().unwrap_or(0),
        notes,
        output_bytes: weights + companions,
    })
}

#[cfg(test)]
#[path = "tests/jobs.rs"]
mod tests;
