//! Tests for module `forge::readiness`: the disk checks and the Build button state.

use std::fs;

use super::*;

fn inputs(base: Option<PathBuf>, donor: Option<PathBuf>, out_dir: Option<PathBuf>, name: &str) -> Inputs {
    Inputs { base, donor, out_dir, out_name: name.into() }
}

#[test]
fn checks_same_model_and_output_location() {
    let d = std::env::temp_dir().join(format!("ff-gui-readiness-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    let (base, donor) = (d.join("base"), d.join("donor"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&donor).unwrap();

    let c = check(&inputs(Some(base.clone()), Some(donor.clone()), None, ""));
    assert_eq!(c, Checks { same_model: false, output: None, disk: None });
    let c = check(&inputs(Some(base.clone()), Some(base.join("..").join("base")), None, ""));
    assert!(c.same_model);
    let c = check(&inputs(Some(base.clone()), Some(donor.clone()), Some(d.clone()), "a/b"));
    assert_eq!(c.output, Some(Err(OutputProblem::BadName)));
    let c = check(&inputs(Some(base.clone()), Some(donor.clone()), Some(donor.clone()), "x"));
    assert_eq!(c.output, Some(Err(OutputProblem::InsideSource)));
    let c = check(&inputs(Some(base), Some(donor), Some(d.clone()), "forged"));
    assert_eq!(c.output, Some(Ok(())));
    // The free space of the output volume is read with the other disk checks.
    assert!(matches!(c.disk, Some(Known::Known(free)) if free > 0), "{:?}", c.disk);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn the_build_disk_preflight_counts_the_partial_copy_and_a_reserve() {
    let dir = Path::new("out");
    let gib = 1024 * 1024 * 1024;
    assert_eq!(disk_status(gib, dir, &Known::Known(10 * gib)).0, PreflightStatus::Fits);
    assert_eq!(
        disk_status(gib, dir, &Known::Known(gib)).0,
        PreflightStatus::Insufficient,
        "no room for the reserve"
    );
    assert_eq!(disk_status(gib, dir, &Known::unknown("not read")).0, PreflightStatus::Unknown);
    assert!(disk_status(gib, dir, &Known::Known(10 * gib)).1 > gib, "need includes the reserve");
}

#[test]
fn build_is_clickable_only_when_ready_and_idle() {
    assert!(build_enabled(&[], false));
    assert!(!build_enabled(&[], true), "a second click while building must do nothing");
    assert!(!build_enabled(&[Problem::NeedBase], false));
    assert!(!build_enabled(&[Problem::Output(OutputProblem::InsideSource)], false));
}

#[test]
fn every_problem_key_is_listed() {
    let all = [
        Problem::NeedBase,
        Problem::NeedDonor,
        Problem::SameModel,
        Problem::NeedAnalysis,
        Problem::AnalysisFailed,
        Problem::Tokenizer,
        Problem::NeedBlocks,
        Problem::NeedPreview,
        Problem::PreviewFailed,
        Problem::NeedOutput,
        Problem::Output(OutputProblem::BadName),
        Problem::Output(OutputProblem::Exists(PathBuf::new())),
        Problem::Output(OutputProblem::InsideSource),
        Problem::Output(OutputProblem::Unresolved(String::new())),
        Problem::Output(OutputProblem::NoSpace { need: 2, available: 1 }),
    ];
    assert_eq!(all.map(|p| p.message_key()), Problem::KEYS);
}
