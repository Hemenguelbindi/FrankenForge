//! Tests for module `transplant`.

use super::*;
use crate::testutil::{Fake, model};

#[test]
fn replaces_matching_block() {
    let mut plan = Plan::from_base("a", &model(&[("l.layers.0.w", &[2, 2]), ("l.layers.0.b", &[2])]));
    let donor = Fake(model(&[("x.layers.5.w", &[2, 2]), ("x.layers.5.b", &[2])]));
    apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.", false).unwrap();
    assert_eq!(plan.provenance()["b"], 2);
    assert_eq!(
        plan.tensors["l.layers.0.w"].payload,
        Payload::Source { alias: "b".into(), name: "x.layers.5.w".into() }
    );
}

#[test]
fn rejects_shape_mismatch() {
    let mut plan = Plan::from_base("a", &model(&[("l.layers.0.w", &[2, 2])]));
    let donor = Fake(model(&[("x.layers.5.w", &[3, 2])]));
    let err = apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.", false).unwrap_err();
    assert!(matches!(err, Error::Incompatible(_)));
}

#[test]
fn leftover_base_tensors_fail_unless_keep_extra() {
    let base = model(&[("l.layers.0.w", &[2]), ("l.layers.0.extra", &[2])]);
    let donor = Fake(model(&[("x.layers.5.w", &[2])]));

    let mut plan = Plan::from_base("a", &base);
    let err = apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.", false).unwrap_err();
    assert!(
        matches!(&err, Error::Incompatible(m) if m.contains("l.layers.0.extra") && m.contains("keep_extra")),
        "{err}"
    );
    assert_eq!(plan.provenance().get("b"), None, "a failed step changes nothing");

    apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.", true).unwrap();
    assert_eq!(plan.provenance()["b"], 1);
    assert_eq!(plan.provenance()["a"], 1);
}

#[test]
fn new_block_must_match_the_stack() {
    let base = model(&[("l.layers.0.w", &[2, 2]), ("l.layers.1.w", &[2, 2])]);
    let mut plan = Plan::from_base("a", &base);
    let bad = Fake(model(&[("x.layers.0.w", &[3, 2])]));
    let err = apply(&mut plan, &bad, "b", "x.layers.0.", "l.layers.2.", false).unwrap_err();
    assert!(matches!(&err, Error::Incompatible(m) if m.contains("new block")), "{err}");

    let good = Fake(model(&[("x.layers.0.w", &[2, 2])]));
    apply(&mut plan, &good, "b", "x.layers.0.", "l.layers.2.", false).unwrap();
    assert!(plan.tensors.contains_key("l.layers.2.w"));
}
