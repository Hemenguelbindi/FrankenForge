//! Tests for module `transplant`.

use super::*;
use crate::testutil::{Fake, model};

#[test]
fn replaces_matching_block() {
    let mut plan = Plan::from_base("a", &model(&[("l.layers.0.w", &[2, 2]), ("l.layers.0.b", &[2])]));
    let donor = Fake(model(&[("x.layers.5.w", &[2, 2]), ("x.layers.5.b", &[2])]));
    apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.").unwrap();
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
    let err = apply(&mut plan, &donor, "b", "x.layers.5.", "l.layers.0.").unwrap_err();
    assert!(matches!(err, Error::Incompatible(_)));
}
