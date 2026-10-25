//! Tests for module `plan`.

use super::*;
use crate::testutil::model;

#[test]
fn compact_closes_gaps_in_numeric_order() {
    let base = model(&[
        ("m.layers.0.w", &[1]),
        ("m.layers.2.w", &[2]),
        ("m.layers.10.w", &[3]),
        ("v.blocks.3.w", &[4]),
        ("lm_head.weight", &[5]),
    ]);
    let mut plan = Plan::from_base("a", &base);
    plan.compact();
    let names: Vec<&str> = plan.tensors.keys().map(String::as_str).collect();
    assert_eq!(names, ["lm_head.weight", "m.layers.0.w", "m.layers.1.w", "m.layers.2.w", "v.blocks.0.w"]);
    // layers.10 became layers.2, but its bytes still come from the original name.
    assert_eq!(plan.tensors["m.layers.2.w"].info.shape, [3]);
    assert_eq!(
        plan.tensors["m.layers.2.w"].payload,
        Payload::Source { alias: "a".into(), name: "m.layers.10.w".into() }
    );
    assert!(plan.to_model().layout().gaps().is_empty());
}

#[test]
fn blocks_changed_tracks_composition_order_and_origin() {
    let base = model(&[("m.layers.0.w", &[2]), ("m.layers.1.w", &[2]), ("head", &[1])]);
    let fresh = || Plan::from_base("a", &base);
    assert!(!fresh().blocks_changed("a", &base), "no-op");

    // A tensor outside the stacks does not affect layer structure.
    let mut plan = fresh();
    plan.tensors.get_mut("head").unwrap().payload =
        Payload::Source { alias: "b".into(), name: "head".into() };
    assert!(!plan.blocks_changed("a", &base));

    // Different origin of a block at the same index and block count.
    let mut plan = fresh();
    plan.tensors.get_mut("m.layers.0.w").unwrap().payload =
        Payload::Source { alias: "b".into(), name: "x.layers.5.w".into() };
    assert!(plan.blocks_changed("a", &base));

    // Swapped blocks: same count, same names, sources exchanged.
    let mut plan = fresh();
    let p = &mut plan.tensors;
    let (x, y) = (p["m.layers.0.w"].payload.clone(), p["m.layers.1.w"].payload.clone());
    p.get_mut("m.layers.0.w").unwrap().payload = y;
    p.get_mut("m.layers.1.w").unwrap().payload = x;
    assert!(plan.blocks_changed("a", &base));

    // Removed block.
    let mut plan = fresh();
    plan.tensors.remove("m.layers.1.w");
    assert!(plan.blocks_changed("a", &base));
}

#[test]
fn generated_tensor_marks_plan_experimental() {
    let mut plan = Plan::from_base("a", &model(&[("w", &[2])]));
    assert_eq!(plan.experimental(), None);

    let info = TensorInfo { name: "bridge.weight".into(), dtype: ff_core::DType::F32, shape: vec![2, 2] };
    plan.insert(Planned { info, payload: Payload::Generated(AdapterInit::Identity) });
    let e = plan.experimental().unwrap();
    assert!(e.experimental);
    assert_eq!(e.runtime_support, "missing");
    assert_eq!(e.numerical_validation, "not_performed");
    assert_eq!(e.quality_validation, "not_performed");
    assert_eq!(e.tensors, ["bridge.weight"]);
    assert!(e.text().contains("runtime_support: missing"));
}
