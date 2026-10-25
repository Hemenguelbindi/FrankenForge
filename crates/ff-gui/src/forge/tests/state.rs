//! Tests for module `forge::state`: selection rules, stale results, the generated recipe, the
//! readiness list and how often its disk checks run.

use std::cell::Cell;
use std::fs;

use ff_surgery::compare::{BlockDiff, Report, StackDiff, TensorDiff, TokenizerDiff, TokenizerMatch};

use super::*;
use crate::failure::Failure;
use crate::forge::build::Built;
use crate::forge::paths::OutputProblem;

/// Base has blocks 0..=3, Donor 0..=2 and 4: 0 and 2 fit, 1 differs, 3 is Base-only, 4 Donor-only.
fn report(tokenizer: TokenizerMatch) -> Report {
    Report {
        a: "a".into(),
        b: "b".into(),
        tensors: TensorDiff::default(),
        stacks: vec![StackDiff {
            stack: "model.layers".into(),
            blocks_a: 4,
            blocks_b: 4,
            compatible: vec![0, 2],
            incompatible: vec![BlockDiff { index: 1, differences: vec!["mlp.w: F32 [2] vs F32 [3]".into()] }],
            only_a: vec![3],
            only_b: vec![4],
        }],
        tokenizer: TokenizerDiff { status: tokenizer, vocab_a: Some(10), vocab_b: Some(10) },
    }
}

fn block(index: usize) -> BlockId {
    BlockId { stack: "model.layers".into(), index }
}

/// Base and Donor chosen and analyzed (identical tokenizers).
fn analyzed(base: &str, donor: &str) -> Forge {
    let mut f = Forge::default();
    f.set_base(base.into());
    f.set_donor(donor.into());
    f.apply_analysis(f.pair().unwrap(), Ok(Analysis::new(report(TokenizerMatch::Identical))));
    f
}

fn preview(from_base: usize, from_donor: usize) -> Preview {
    Preview { from_base, from_donor, notes: Vec::new(), output_bytes: 1024 }
}

/// Disk checks stand-in: distinct models, output location fine once chosen.
fn fine(i: &Inputs) -> Checks {
    Checks { same_model: false, output: i.out_dir.as_ref().map(|_| Ok(())), disk: None }
}

/// Everything ready except that the output volume reports `free`.
fn ready_with_disk(free: Known<u64>) -> (Forge, Vec<Problem>) {
    let mut f = analyzed("b", "d");
    f.toggle(&block(0));
    let sel = f.selection().unwrap();
    let gib = 1024 * 1024 * 1024;
    f.apply_preview(sel, Ok(Preview { from_base: 7, from_donor: 1, notes: vec![], output_bytes: gib }));
    f.set_out_dir("out".into());
    f.set_out_name("x".into());
    let problems = f.problems_with(|i| Checks { disk: Some(free), ..fine(i) });
    (f, problems)
}

#[test]
fn a_known_disk_shortage_blocks_the_build() {
    let (f, problems) = ready_with_disk(Known::Known(512 * 1024 * 1024));
    assert!(
        matches!(problems[..], [Problem::Output(OutputProblem::NoSpace { available, .. })] if available == 512 * 1024 * 1024),
        "{problems:?}"
    );
    assert_eq!(f.disk().unwrap().0, PreflightStatus::Insufficient);
}

#[test]
fn unknown_free_space_is_shown_but_does_not_claim_a_fit() {
    let (f, problems) = ready_with_disk(Known::unknown("volume not found"));
    assert!(problems.is_empty(), "unknown space does not block; the core still fails closed");
    let (status, _, available) = f.disk().unwrap();
    assert_eq!(status, PreflightStatus::Unknown);
    assert!(matches!(available, Known::Unknown(_)));
    let (f, problems) = ready_with_disk(Known::Known(100 * 1024 * 1024 * 1024));
    assert!(problems.is_empty());
    assert_eq!(f.disk().unwrap().0, PreflightStatus::Fits);
}

#[test]
fn only_structurally_compatible_blocks_can_be_selected() {
    let mut f = analyzed("b", "d");
    for i in 0..5 {
        f.toggle(&block(i));
    }
    assert_eq!(f.selection().unwrap().blocks, [block(0), block(2)]);
    // Different structure (1), Base-only (3) and Donor-only (4) stay unselected.
    for i in [1, 3, 4] {
        assert!(!f.is_selected(&block(i)), "block {i}");
    }
    f.toggle(&block(0));
    assert_eq!(f.selection().unwrap().blocks, [block(2)]);
    f.clear_selection();
    assert!(f.selection().is_none());
}

#[test]
fn nothing_is_selectable_before_the_analysis_arrives() {
    let mut f = Forge::default();
    f.set_base("b".into());
    f.set_donor("d".into());
    f.toggle(&block(0));
    assert!(f.selection().is_none());
}

#[test]
fn choosing_another_model_drops_analysis_selection_and_preview() {
    let mut f = analyzed("b", "d");
    f.toggle(&block(0));
    let sel = f.selection().unwrap();
    f.apply_preview(sel, Ok(preview(1, 1)));
    f.set_donor("other".into());
    assert!(f.analysis().is_none() && f.selection().is_none() && f.preview().is_none());
}

#[test]
fn a_result_for_an_old_pair_or_selection_is_dropped() {
    let mut f = Forge::default();
    f.set_base("b".into());
    f.set_donor("d".into());
    let old = f.pair().unwrap();
    f.set_donor("d2".into());
    f.apply_analysis(old, Ok(Analysis::new(report(TokenizerMatch::Identical))));
    assert!(f.analysis().is_none());

    let mut f = analyzed("b", "d");
    f.toggle(&block(0));
    let old = f.selection().unwrap();
    f.toggle(&block(2));
    f.apply_preview(old, Ok(preview(1, 1)));
    assert!(f.preview().is_none());
}

#[test]
fn a_preview_starts_once_per_selection() {
    let mut f = analyzed("b", "d");
    assert_eq!(f.preview_needed(None), None, "no blocks, no preview");
    f.toggle(&block(2));
    let sel = f.preview_needed(None).unwrap();
    assert_eq!(f.preview_needed(Some(&sel)), None, "already running");
    f.apply_preview(sel.clone(), Err(Failure::crashed()));
    assert_eq!(f.preview_needed(None), None, "a failed preview is shown, not retried in a loop");
    f.toggle(&block(0));
    assert!(f.preview_needed(Some(&sel)).is_some(), "a new selection needs its own preview");
}

#[test]
fn the_recipe_moves_each_block_to_the_same_index() {
    let mut f = analyzed("models/base", "models/donor");
    f.toggle(&block(2));
    f.toggle(&block(0));
    let r = recipe(&f.selection().unwrap());
    assert_eq!(r.base, BASE);
    assert_eq!(r.sources[BASE], PathBuf::from("models/base"));
    assert_eq!(r.sources[DONOR], PathBuf::from("models/donor"));
    let steps: Vec<(String, String, String)> = r
        .steps
        .iter()
        .map(|s| match s {
            Step::Transplant { from, src, dst } => (from.clone(), src.clone(), dst.clone()),
            other => panic!("unexpected step {other:?}"),
        })
        .collect();
    assert_eq!(
        steps,
        [
            (DONOR.into(), "model.layers.0.".into(), "model.layers.0.".into()),
            (DONOR.into(), "model.layers.2.".into(), "model.layers.2.".into()),
        ]
    );
}

#[test]
fn readiness_lists_what_is_missing_in_order() {
    let mut f = Forge::default();
    assert_eq!(f.problems_with(fine), [Problem::NeedBase, Problem::NeedOutput]);

    let mut f = analyzed("b", "d");
    assert_eq!(f.problems_with(fine), [Problem::NeedBlocks, Problem::NeedOutput]);
    f.toggle(&block(0));
    assert_eq!(f.problems_with(fine), [Problem::NeedPreview, Problem::NeedOutput]);
    let sel = f.selection().unwrap();
    f.apply_preview(sel.clone(), Err(Failure::crashed()));
    assert_eq!(f.problems_with(fine), [Problem::PreviewFailed, Problem::NeedOutput]);
    f.apply_preview(sel, Ok(preview(1, 1)));
    f.out_dir = Some("out".into());
    assert_eq!(f.problems_with(fine), []);
}

#[test]
fn non_identical_tokenizers_block_the_build() {
    for status in [TokenizerMatch::SameVocab, TokenizerMatch::DifferentVocab, TokenizerMatch::Missing] {
        let mut f = Forge::default();
        f.set_base("b".into());
        f.set_donor("d".into());
        f.apply_analysis(f.pair().unwrap(), Ok(Analysis::new(report(status))));
        assert!(f.problems_with(fine).contains(&Problem::Tokenizer), "{status:?}");
    }
}

#[test]
fn disk_checks_run_only_when_their_inputs_change() {
    let calls = Cell::new(0);
    let counted = |i: &Inputs| {
        calls.set(calls.get() + 1);
        fine(i)
    };
    let mut f = analyzed("b", "d");
    f.toggle(&block(0));
    for _ in 0..5 {
        f.problems_with(counted);
    }
    assert_eq!(calls.get(), 1, "repeated frames with the same paths must not touch the disk");

    // Selection and preview are not disk inputs.
    f.toggle(&block(2));
    f.problems_with(counted);
    assert_eq!(calls.get(), 1);

    f.out_dir = Some("out".into());
    f.problems_with(counted);
    f.out_name = "x".into();
    f.problems_with(counted);
    f.set_donor("d2".into());
    f.problems_with(counted);
    f.problems_with(counted);
    assert_eq!(calls.get(), 4, "each changed input re-checks exactly once");
}

/// Everything chosen, previewed and checked (with `fine`): ready to build.
fn ready() -> Forge {
    let mut f = analyzed("b", "d");
    f.toggle(&block(0));
    let sel = f.selection().unwrap();
    f.apply_preview(sel, Ok(preview(7, 1)));
    f.set_out_dir("out".into());
    f.set_out_name("forged".into());
    f
}

fn built(req: &BuildRequest) -> Built {
    Built {
        out: req.out.clone(),
        recipe: "r".into(),
        lock: "l".into(),
        log: Some("g".into()),
        tensors: 8,
        params: 8,
        planned: req.preview.clone(),
        warnings: Vec::new(),
    }
}

#[test]
fn a_build_starts_only_when_ready() {
    let mut f = analyzed("b", "d");
    assert_eq!(f.begin_build_with(fine), None, "no blocks yet");
    assert!(!f.running());

    let mut f = ready();
    let req = f.begin_build_with(fine).unwrap();
    assert_eq!(req.out, PathBuf::from("out").join("forged"));
    assert_eq!(req.selection.blocks, [block(0)]);
    assert_eq!((req.preview.from_base, req.preview.from_donor), (7, 1));
    assert!(f.running());
    assert_eq!(f.begin_build_with(fine), None, "a second click while building does nothing");
}

#[test]
fn every_input_is_locked_while_building() {
    let mut f = ready();
    let req = f.begin_build_with(fine).unwrap();
    f.set_base("other-base".into());
    f.set_donor("other-donor".into());
    f.toggle(&block(2));
    f.toggle(&block(0));
    f.clear_selection();
    f.set_out_dir("elsewhere".into());
    f.set_out_name("renamed".into());
    assert_eq!(f.selection().unwrap(), req.selection, "models and blocks unchanged");
    assert_eq!(f.out().unwrap(), req.out, "output unchanged");
    assert!(f.running());
}

#[test]
fn a_finished_build_is_shown_until_an_input_changes() {
    let mut f = ready();
    let req = f.begin_build_with(fine).unwrap();
    let mut stale = req.clone();
    stale.out = "old".into();
    f.finish_build(stale, Ok(Ok(built(&req))));
    assert!(f.running(), "a result for another request is ignored");

    f.finish_build(req.clone(), Ok(Ok(built(&req))));
    assert!(matches!(f.build_status(), BuildStatus::Done(Ok(b)) if b.out == req.out));
    // The output now exists on disk: the next readiness check must look again.
    let calls = Cell::new(0);
    f.problems_with(|i| {
        calls.set(calls.get() + 1);
        fine(i)
    });
    assert_eq!(calls.get(), 1);

    f.set_out_name("next".into());
    assert!(matches!(f.build_status(), BuildStatus::Idle));
}

#[test]
fn a_failed_or_crashed_build_is_never_shown_as_built() {
    let mut f = ready();
    let req = f.begin_build_with(fine).unwrap();
    let failure = Failure::from_error(&ff_core::Error::Verify("w: content hash mismatch".into()));
    f.finish_build(
        req,
        Ok(Err(BuildError::Failed { failure, recipe: Some("r".into()), log: Some("g".into()) })),
    );
    assert!(matches!(f.build_status(), BuildStatus::Done(Err(BuildError::Failed { .. }))));
    assert!(!f.running());

    let mut f = ready();
    let req = f.begin_build_with(fine).unwrap();
    f.finish_build(req, Err(Failure::crashed()));
    assert!(matches!(
        f.build_status(),
        BuildStatus::Done(Err(BuildError::Failed { recipe: None, log: None, .. }))
    ));
}

#[test]
fn the_real_checks_report_the_output_problems() {
    let d = std::env::temp_dir().join(format!("ff-gui-state-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    let (base, donor) = (d.join("base"), d.join("donor"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&donor).unwrap();

    let mut f = analyzed(base.to_str().unwrap(), donor.to_str().unwrap());
    f.toggle(&block(2));
    let sel = f.selection().unwrap();
    f.apply_preview(sel, Ok(preview(7, 1)));
    f.out_dir = Some(d.clone());
    f.out_name = "bad/name".into();
    assert_eq!(f.problems(), [Problem::Output(OutputProblem::BadName)]);
    f.out_name = "base".into();
    assert_eq!(f.problems(), [Problem::Output(OutputProblem::Exists(base.clone()))]);
    f.out_dir = Some(base.clone());
    f.out_name = "inside".into();
    assert_eq!(f.problems(), [Problem::Output(OutputProblem::InsideSource)]);
    f.out_dir = Some(d.clone());
    f.out_name = "forged".into();
    assert_eq!(f.problems(), []);

    f.set_donor(base.join("..").join("base"));
    assert!(f.problems().contains(&Problem::SameModel));
    fs::remove_dir_all(&d).unwrap();
}
