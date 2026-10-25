//! Tests for module `loader`.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
use crate::failure::FailureKind;

type Loader = super::Loader<PathBuf, ()>;
type Outcome = super::Outcome<()>;

/// Polls until the loader yields something or `limit` passes.
fn wait(loader: &mut Loader, limit: Duration) -> Option<(PathBuf, Outcome)> {
    let start = Instant::now();
    while start.elapsed() < limit {
        if let Some(done) = loader.poll() {
            return Some(done);
        }
        thread::sleep(Duration::from_millis(5));
    }
    None
}

fn failed(detail: &str) -> Outcome {
    Err(Failure { kind: FailureKind::Unknown, detail: detail.into() })
}

fn detail(outcome: Outcome) -> String {
    outcome.expect_err("these test openers always fail").detail
}

#[test]
fn superseded_result_is_never_applied() {
    let mut loader = Loader::default();
    let (release_slow, gate) = mpsc::channel::<()>();
    loader.start(
        "slow".into(),
        move |_| {
            gate.recv().ok();
            failed("slow")
        },
        || {},
    );
    assert_eq!(loader.pending().map(PathBuf::as_path), Some(Path::new("slow")));

    // A second open replaces the first while it is still running.
    loader.start("fast".into(), |_| failed("fast"), || {});
    let (path, outcome) = wait(&mut loader, Duration::from_secs(5)).expect("fast request finishes");
    assert_eq!(path, Path::new("fast"));
    assert_eq!(detail(outcome), "fast");

    // The slow one finishing later is dropped, not delivered.
    release_slow.send(()).unwrap();
    assert!(wait(&mut loader, Duration::from_millis(200)).is_none());
    assert_eq!(loader.pending(), None);
}

#[test]
fn cleared_request_is_never_applied() {
    let mut loader = Loader::default();
    let (release, gate) = mpsc::channel::<()>();
    loader.start(
        "tensor".into(),
        move |_| {
            gate.recv().ok();
            Ok(())
        },
        || {},
    );
    // The user picks another tensor: the running analysis is forgotten.
    loader.clear();
    release.send(()).unwrap();
    assert!(wait(&mut loader, Duration::from_millis(200)).is_none());
}

#[test]
fn worker_panic_wakes_the_ui_and_becomes_an_error() {
    let mut loader = Loader::default();
    let (woke, woken) = mpsc::channel();
    loader.start("boom".into(), |_| panic!("parser bug"), move || woke.send(()).unwrap());
    woken.recv_timeout(Duration::from_secs(5)).expect("wake called despite the panic");
    // The disconnect is already visible when the UI is woken.
    let (_, outcome) = loader.poll().expect("panic is reported");
    assert_eq!(outcome.err().map(|f| f.kind), Some(FailureKind::Crashed));
}

#[test]
fn wake_runs_after_the_result_is_sent() {
    let mut loader = Loader::default();
    let (woke, woken) = mpsc::channel();
    loader.start("x".into(), |_| failed("done"), move || woke.send(()).unwrap());
    woken.recv_timeout(Duration::from_secs(5)).expect("wake called");
    // After wake the result must already be there: no extra frame is needed to see it.
    assert!(loader.poll().is_some());
}
