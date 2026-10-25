//! Tests for module `compare`.

use super::*;
use crate::testutil::model;

fn pair() -> (Model, Model) {
    let a = model(&[("m.layers.0.w", &[2]), ("m.layers.1.w", &[2]), ("m.layers.2.w", &[2]), ("head", &[1])]);
    let b = model(&[
        ("m.layers.0.w", &[2]),
        ("m.layers.1.w", &[4]),
        ("m.layers.1.extra", &[1]),
        ("v.blocks.0.w", &[1]),
        ("head", &[1]),
    ]);
    (a, b)
}

#[test]
fn tensor_diff_buckets() {
    let (a, b) = pair();
    let d = tensors(&a, &b);
    assert_eq!(d.same, 2);
    assert_eq!(d.mismatched.len(), 1);
    assert_eq!(d.mismatched[0].name, "m.layers.1.w");
    assert_eq!(d.mismatched[0].b.shape, [4]);
    assert_eq!(d.only_a, ["m.layers.2.w"]);
    assert_eq!(d.only_b, ["m.layers.1.extra", "v.blocks.0.w"]);
}

#[test]
fn stacks_pair_blocks_by_index() {
    let (a, b) = pair();
    let s = stacks(&a, &b);
    assert_eq!(s.len(), 2);
    let m = &s[0];
    assert_eq!((m.stack.as_str(), m.blocks_a, m.blocks_b), ("m.layers", 3, 2));
    assert_eq!(m.compatible, [0]);
    assert_eq!(m.incompatible.len(), 1);
    assert_eq!(m.incompatible[0].index, 1);
    assert_eq!(m.incompatible[0].differences, ["w: F32 [2] vs F32 [4]", "extra: only in B"]);
    assert_eq!((m.only_a.as_slice(), m.only_b.as_slice()), (&[2][..], &[][..]));
    // The stack exists only in B: no pairs, no compatible blocks, its block is listed as only in B.
    assert_eq!((s[1].blocks_a, s[1].blocks_b, s[1].compatible.len()), (0, 1, 0));
    assert_eq!((s[1].only_a.as_slice(), s[1].only_b.as_slice()), (&[][..], &[0][..]));
}

#[test]
fn every_index_of_either_model_is_reported_once() {
    // A has blocks 0, 1, 5; B has 1, 2, 3, 5 (gaps and different counts on both sides).
    let a = model(&[("m.layers.0.w", &[1]), ("m.layers.1.w", &[1]), ("m.layers.5.w", &[1])]);
    let b = model(&[
        ("m.layers.1.w", &[1]),
        ("m.layers.2.w", &[1]),
        ("m.layers.3.w", &[1]),
        ("m.layers.5.w", &[2]),
    ]);
    let s = &stacks(&a, &b)[0];
    assert_eq!((s.blocks_a, s.blocks_b), (3, 4));
    assert_eq!(s.compatible, [1]);
    assert_eq!(s.incompatible.iter().map(|d| d.index).collect::<Vec<_>>(), [5]);
    assert_eq!(s.only_a, [0]);
    assert_eq!(s.only_b, [2, 3]);
}

#[test]
fn text_and_json_reports() {
    let (a, b) = pair();
    let report = Report {
        a: "A".into(),
        b: "B".into(),
        tensors: tensors(&a, &b),
        stacks: stacks(&a, &b),
        tokenizer: TokenizerDiff { status: TokenizerMatch::Missing, vocab_a: None, vocab_b: None },
    };
    let text = report.text();
    assert!(text.contains("tensors: 2 same, 1 mismatched, 1 only in A, 2 only in B"), "{text}");
    assert!(text.contains("m.layers: A 3 blocks, B 2 blocks; 1 compatible, 1 incompatible"), "{text}");
    assert!(text.contains("tokenizer: Missing"), "{text}");
    assert!(text.contains("only in A:\n    [2]"), "{text}");

    let json: serde_json::Value = serde_json::from_str(&report.json()).unwrap();
    assert_eq!(json["tokenizer"]["status"], "missing");
    assert_eq!(json["stacks"][0]["compatible"], serde_json::json!([0]));
    assert_eq!(json["stacks"][0]["only_a"], serde_json::json!([2]));
    assert_eq!(json["tensors"]["mismatched"][0]["a"]["dtype"], "F32");
}

#[test]
fn text_lists_are_truncated() {
    let items: Vec<String> = (0..TEXT_LIMIT + 2).map(|i| i.to_string()).collect();
    let mut out = String::new();
    list(&mut out, "x", &items);
    assert!(out.contains("... and 2 more"), "{out}");
}
