//! Tests for module `report`: one test per block status, removed blocks and tensors, outside
//! tensors, and the JSON schema.

use std::collections::BTreeMap;

use ff_core::TensorSource;

use super::*;
use crate::testutil::{Fake, model};
use crate::{AdapterInit, Sources, Step, ops};

fn layers(blocks: usize) -> Model {
    let names: Vec<(String, Vec<usize>)> = (0..blocks)
        .flat_map(|i| [(format!("m.layers.{i}.w"), vec![2, 2]), (format!("m.layers.{i}.b"), vec![2])])
        .chain([("head".to_owned(), vec![2])])
        .collect();
    let refs: Vec<(&str, &[usize])> = names.iter().map(|(n, s)| (n.as_str(), s.as_slice())).collect();
    model(&refs)
}

/// Base `a` and donor `b`.
pub(super) fn pair(base: Model, donor: Model) -> Sources {
    let map: BTreeMap<String, Box<dyn TensorSource>> = BTreeMap::from([
        ("a".to_owned(), Box::new(Fake(base)) as Box<dyn TensorSource>),
        ("b".to_owned(), Box::new(Fake(donor)) as Box<dyn TensorSource>),
    ]);
    Sources::new(map)
}

fn transplant(src: &str, dst: &str, keep_extra: bool) -> Step {
    Step::Transplant { from: "b".into(), src: src.into(), dst: dst.into(), keep_extra }
}

fn report_of(s: &Sources, steps: &[Step]) -> Report {
    let base = s.get("a").unwrap().model();
    let mut plan = Plan::from_base("a", base);
    for step in steps {
        ops::apply(&mut plan, step, s).unwrap();
    }
    plan.compact();
    Report::new("a", base, &plan)
}

fn report(blocks: usize, steps: &[Step]) -> Report {
    report_of(&pair(layers(blocks), layers(blocks)), steps)
}

fn statuses(r: &Report) -> Vec<Status> {
    r.stacks[0].blocks.iter().map(|b| b.status.clone()).collect()
}

#[test]
fn untouched_plan_is_unchanged_everywhere() {
    let r = report(2, &[]);
    assert_eq!(statuses(&r), [Status::Unchanged, Status::Unchanged]);
    assert!(r.stacks[0].blocks.iter().all(|b| b.replaced.is_empty() && b.delta.is_none()));
    assert_eq!((r.stacks[0].blocks_before, r.stacks[0].blocks_after), (2, 2));
    assert!(r.removed_blocks.is_empty() && r.removed_tensors.is_empty());
    assert_eq!(r.outside.len(), 1);
    assert!(r.outside[0].unchanged && r.outside[0].delta.is_none());
    assert_eq!(r.model_behavior, NOT_EVALUATED);
    assert_eq!(r.file, FileCheck::NotWritten);
}

#[test]
fn donor_block_is_replaced_with_pending_delta() {
    let r = report(3, &[transplant("m.layers.2.", "m.layers.1.", false)]);
    assert_eq!(
        statuses(&r),
        [Status::Unchanged, Status::Replaced { alias: "b".into(), from: 2 }, Status::Unchanged]
    );
    let b = &r.stacks[0].blocks[1];
    assert_eq!(b.replaced.len(), 2);
    assert_eq!(b.delta, Some(Delta::NotComputed));
    assert_eq!(b.sources, [BlockSource { alias: "b".into(), index: Some(2) }]);
    assert!(r.removed_blocks.is_empty(), "a replaced block is not a removed one");
    assert!(r.removed_tensors.is_empty());
}

#[test]
fn removal_shifts_later_blocks_and_lists_the_removed_one() {
    let r = report(3, &[Step::Remove { select: "m.layers.0.".into() }]);
    assert_eq!(statuses(&r), [Status::Shifted { from: 1 }, Status::Shifted { from: 2 }]);
    assert_eq!(r.removed_blocks, [BlockRef { stack: "m.layers".into(), index: 0 }]);
    assert_eq!((r.stacks[0].blocks_before, r.stacks[0].blocks_after), (3, 2));
}

#[test]
fn keep_extra_gives_partial_replacement() {
    let donor = model(&[("m.layers.0.w", &[2, 2])]);
    let r = report_of(&pair(layers(1), donor), &[transplant("m.layers.0.", "m.layers.0.", true)]);
    assert_eq!(statuses(&r), [Status::PartiallyReplaced { replaced: 1, total: 2 }]);
    assert_eq!(
        r.stacks[0].blocks[0].sources,
        [
            BlockSource { alias: "a".into(), index: Some(0) },
            BlockSource { alias: "b".into(), index: Some(0) }
        ]
    );
    assert_eq!(r.stacks[0].blocks[0].replaced.len(), 1);
}

#[test]
fn block_past_the_base_end_is_added() {
    let r = report(2, &[transplant("m.layers.0.", "m.layers.2.", false)]);
    assert_eq!(statuses(&r), [Status::Unchanged, Status::Unchanged, Status::Added]);
    assert_eq!((r.stacks[0].blocks_before, r.stacks[0].blocks_after), (2, 3));
    assert!(r.stacks[0].blocks[2].replaced.is_empty(), "nothing in base to compare with");
}

#[test]
fn generated_tensor_marks_block_and_experimental() {
    let step = Step::Adapter {
        name: "m.layers.0.adapter".into(),
        in_dim: 2,
        out_dim: 2,
        init: AdapterInit::Identity,
        dtype: ff_core::DType::F32,
    };
    let r = report(1, &[step]);
    assert_eq!(statuses(&r), [Status::Generated]);
    assert!(r.experimental.is_some());
}

#[test]
fn removed_tensors_and_replaced_outside_tensor() {
    let r = report(2, &[Step::Remove { select: "m.layers.1.b".into() }, transplant("head", "head", false)]);
    assert_eq!(r.removed_tensors, ["m.layers.1.b"]);
    assert!(r.removed_blocks.is_empty());
    let head = &r.outside[0];
    assert!(!head.unchanged);
    assert_eq!(head.origin, Origin::Source { alias: "b".into(), name: "head".into() });
    assert_eq!(head.delta, Some(Delta::NotComputed));
}

#[test]
fn json_schema_is_stable_and_round_trips() {
    let r = report(2, &[transplant("m.layers.1.", "m.layers.1.", false)]);
    let v: serde_json::Value = serde_json::from_str(&r.json()).unwrap();
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "base",
            "config_changes",
            "config_present",
            "experimental",
            "file",
            "model_behavior",
            "outside",
            "removed_blocks",
            "removed_tensors",
            "schema_version",
            "stacks",
            "warnings"
        ]
    );
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["model_behavior"], "not_evaluated");
    assert_eq!(v["file"], "not_written");
    let block = &v["stacks"][0]["blocks"][1];
    assert_eq!(block["status"], serde_json::json!({"kind": "replaced", "alias": "b", "from": 1}));
    assert_eq!(block["delta"], serde_json::json!({"kind": "not_computed"}));
    assert_eq!(v["stacks"][0]["blocks"][0]["status"], serde_json::json!({"kind": "unchanged"}));
    assert_eq!(
        v["outside"][0]["origin"],
        serde_json::json!({"kind": "source", "alias": "a", "name": "head"})
    );
    let back: Report = serde_json::from_str(&r.json()).unwrap();
    assert_eq!(back, r);
}

#[test]
fn report_paths_sit_next_to_the_output() {
    let [json, md] = Report::paths_for(Path::new("out/model"));
    assert_eq!(json, Path::new("out/model.report.json"));
    assert_eq!(md, Path::new("out/model.report.md"));
}
