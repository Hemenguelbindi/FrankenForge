//! Tests for module `forge::build_view`: the status never claims stages the core did not report.

use super::*;
use crate::failure::Failure;
use crate::forge::state::{Pair, Preview, Selection};

fn request() -> super::super::build::BuildRequest {
    super::super::build::BuildRequest {
        selection: Selection { pair: Pair { base: "b".into(), donor: "d".into() }, blocks: vec![] },
        out: "out".into(),
        preview: Preview {
            from_base: 1,
            from_donor: 1,
            notes: vec![],
            output_bytes: 0,
            report: crate::forge::state::empty_report(),
        },
    }
}

#[test]
fn a_running_build_claims_no_stage() {
    let s = status(&BuildStatus::Running(request()));
    assert_eq!(s, Status::Running);
    assert!(completed_stages(s).is_empty());
}

#[test]
fn only_success_lists_plan_copy_verify_and_lock_as_done() {
    let built = Built {
        out: "out".into(),
        recipe: "r".into(),
        lock: "l".into(),
        log: None,
        tensors: 1,
        params: 1,
        planned: request().preview,
        warnings: vec![],
        report: crate::forge::state::empty_report(),
    };
    let s = status(&BuildStatus::Done(Ok(built)));
    assert_eq!(s, Status::Succeeded);
    assert_eq!(completed_stages(s), STAGES);
}

#[test]
fn a_stopped_or_refused_build_marks_no_stage_whatever_the_error() {
    for failure in [
        Failure::from_error(&ff_core::Error::Recipe("step 1".into())),
        Failure::from_error(&ff_core::Error::Verify("w: content hash mismatch".into())),
        Failure::crashed(),
    ] {
        let s = status(&BuildStatus::Done(Err(BuildError::Failed { failure, recipe: None, log: None })));
        assert_eq!(s, Status::Stopped);
        assert!(completed_stages(s).is_empty());
    }
    let s = status(&BuildStatus::Done(Err(BuildError::Refused(OutputProblem::InsideSource))));
    assert_eq!(s, Status::NotStarted);
    assert!(completed_stages(s).is_empty());
    assert!(completed_stages(Status::Idle).is_empty());
}
