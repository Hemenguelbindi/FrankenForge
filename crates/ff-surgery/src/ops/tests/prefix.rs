//! Tests for module `prefix`.

use super::*;
use crate::testutil::model;

#[test]
fn remove_then_close_gap() {
    let base = model(&[("m.layers.0.w", &[1]), ("m.layers.1.w", &[1]), ("m.layers.2.w", &[1])]);
    let mut plan = Plan::from_base("a", &base);
    remove(&mut plan, "m.layers.1.").unwrap();
    assert_eq!(plan.to_model().layout().gaps().len(), 1);
    rename(&mut plan, "m.layers.2.", "m.layers.1.").unwrap();
    assert!(plan.to_model().layout().gaps().is_empty());
}

#[test]
fn rename_refuses_to_overwrite() {
    let mut plan = Plan::from_base("a", &model(&[("m.layers.0.w", &[1]), ("m.layers.1.w", &[1])]));
    assert!(rename(&mut plan, "m.layers.1.", "m.layers.0.").is_err());
}
