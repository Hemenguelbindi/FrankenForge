//! Tests for module `text`: range folding and the fixed lines of the text report.

use super::super::{
    BlockReport, BlockSource, Delta, FileCheck, NOT_EVALUATED, Report, SCHEMA_VERSION, Stack, Status,
};
use super::*;

fn block(index: usize, alias: &str, from: usize, status: Status, delta: Option<Delta>) -> BlockReport {
    BlockReport {
        index,
        tensors: 9,
        sources: vec![BlockSource { alias: alias.into(), index: Some(from) }],
        status,
        replaced: Vec::new(),
        delta,
    }
}

fn stack(blocks: Vec<BlockReport>) -> Stack {
    Stack { name: "model.layers".into(), blocks_before: 6, blocks_after: blocks.len(), blocks }
}

fn report(blocks: Vec<BlockReport>) -> Report {
    Report {
        schema_version: SCHEMA_VERSION,
        base: "recipient".into(),
        stacks: vec![stack(blocks)],
        removed_blocks: Vec::new(),
        removed_tensors: Vec::new(),
        outside: Vec::new(),
        config_present: true,
        config_changes: Vec::new(),
        warnings: Vec::new(),
        experimental: None,
        file: FileCheck::Verified,
        model_behavior: NOT_EVALUATED.into(),
    }
}

fn replaced(from: usize) -> Status {
    Status::Replaced { alias: "donor".into(), from }
}

fn summary(rows: &[Row]) -> Vec<(String, String)> {
    rows.iter().map(|r| (r.range(), r.source.clone())).collect()
}

#[test]
fn consecutive_same_status_folds_into_ranges() {
    let measured = Some(Delta::Relative { value: 0.25 });
    let s = stack(vec![
        block(0, "recipient", 0, Status::Unchanged, None),
        block(1, "recipient", 1, Status::Unchanged, None),
        block(2, "donor", 2, replaced(2), measured.clone()),
        block(3, "donor", 3, replaced(3), measured),
        block(4, "recipient", 4, Status::Unchanged, None),
        block(5, "recipient", 5, Status::Unchanged, None),
    ]);
    let pairs = summary(&s.rows());
    assert_eq!(
        pairs,
        [
            ("0-1".into(), "recipient".into()),
            ("2".into(), "donor:2".into()),
            ("3".into(), "donor:3".into()),
            ("4-5".into(), "recipient".into())
        ],
        "rows with a measured delta stay one per block"
    );
}

#[test]
fn shifted_and_unmeasured_replaced_runs_fold_with_source_ranges() {
    let s = stack(vec![
        block(0, "recipient", 1, Status::Shifted { from: 1 }, None),
        block(1, "recipient", 2, Status::Shifted { from: 2 }, None),
        block(2, "donor", 7, replaced(7), Some(Delta::NotComputed)),
        block(3, "donor", 8, replaced(8), Some(Delta::NotComputed)),
        // Not a +1 continuation of the donor run.
        block(4, "donor", 2, replaced(2), Some(Delta::NotComputed)),
    ]);
    let rows = s.rows();
    assert_eq!(
        summary(&rows),
        [
            ("0-1".into(), "recipient:1-2".into()),
            ("2-3".into(), "donor:7-8".into()),
            ("4".into(), "donor:2".into())
        ]
    );
    assert_eq!(rows[1].delta, None, "not computed is not shown as a delta");
}

#[test]
fn partial_replacements_never_fold() {
    let partial = Status::PartiallyReplaced { replaced: 1, total: 9 };
    let s = stack(vec![block(0, "donor", 0, partial.clone(), None), block(1, "donor", 1, partial, None)]);
    assert_eq!(s.rows().len(), 2);
}

#[test]
fn text_matches_the_documented_layout() {
    let mut r = report(vec![
        block(0, "recipient", 0, Status::Unchanged, None),
        block(1, "recipient", 1, Status::Unchanged, None),
        block(2, "donor", 2, replaced(2), Some(Delta::Relative { value: 0.1234 })),
    ]);
    r.stacks[0].blocks_before = 3;
    let text = r.text();
    assert_eq!(
        text,
        "model.layers: 3 -> 3 blocks\n\
         \x20 0-1  recipient  unchanged\n\
         \x20 2    donor:2    replaced (9/9 tensors)   weight delta: 12.34%\n\
         config.json: unchanged\n\
         file: all tensors re-read, hashes match\n\
         model behavior: NOT EVALUATED\n"
    );
    assert!(r.markdown().contains("```text\nmodel.layers: 3 -> 3 blocks\n"));
}

#[test]
fn unknown_delta_and_plan_only_file_are_spelled_out() {
    let mut r =
        report(vec![block(0, "donor", 0, replaced(0), Some(Delta::Unknown { reason: "I32".into() }))]);
    r.file = FileCheck::NotWritten;
    r.config_present = false;
    let text = r.text();
    assert!(text.contains("weight delta: unknown (I32)"), "{text}");
    assert!(text.contains("file: not written (plan only)"), "{text}");
    assert!(text.contains("config.json: none in base"), "{text}");
    assert!(text.ends_with("model behavior: NOT EVALUATED\n"), "{text}");
}
