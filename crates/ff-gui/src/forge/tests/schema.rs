//! Tests for module `forge::schema`: lanes keep each model's own length and gaps, links are
//! explicit pairs, transfer categories never promise more than direct transfer, and the layout
//! fits every window width.

use ff_core::{DType, Model, TensorInfo};
use ff_surgery::compare::{self, Report, TensorDiff, TokenizerDiff, TokenizerMatch};

use super::*;

/// One `[width]` tensor per listed block index.
fn model(blocks: &[(usize, usize)]) -> Model {
    let mut m = Model::default();
    for &(index, width) in blocks {
        m.insert(TensorInfo {
            name: format!("model.layers.{index}.w"),
            dtype: DType::F32,
            shape: vec![width],
        });
    }
    m
}

fn uniform(n: usize) -> Model {
    model(&(0..n).map(|i| (i, 2)).collect::<Vec<_>>())
}

fn schema(base: &Model, donor: &Model) -> Vec<StackSchema> {
    let report = Report {
        a: "base".into(),
        b: "donor".into(),
        tensors: TensorDiff::default(),
        stacks: compare::stacks(base, donor),
        tokenizer: TokenizerDiff { status: TokenizerMatch::Identical, vocab_a: None, vocab_b: None },
    };
    stacks(&Analysis::new(report))
}

fn kinds(lane: &Lane) -> Vec<String> {
    lane.cells
        .iter()
        .map(|c| match c {
            Cell::Block { transfer, .. } => format!("{transfer:?}"),
            Cell::Gap { .. } => "Gap".into(),
        })
        .collect()
}

#[test]
fn zero_one_thirty_and_a_hundred_blocks() {
    assert!(schema(&uniform(0), &uniform(0)).is_empty(), "no stacks, nothing to draw");
    for n in [1, 30, 100] {
        let s = &schema(&uniform(n), &uniform(n))[0];
        assert_eq!((s.base.cells.len(), s.donor.cells.len(), s.width()), (n, n, n));
        assert!(kinds(&s.base).iter().all(|k| k == "Direct"));
    }
}

#[test]
fn each_lane_keeps_its_own_length() {
    let s = &schema(&uniform(30), &uniform(24))[0];
    assert_eq!((s.base.cells.len(), s.donor.cells.len(), s.width()), (30, 24, 30));
    // Base blocks 24..30 have no partner: they need a mapping, the Donor lane just ends.
    assert_eq!(kinds(&s.base)[24..], ["NeedsMapping"; 6]);
    assert_eq!(kinds(&s.donor)[23], "Direct");
}

#[test]
fn gaps_in_numbering_are_shown_not_closed() {
    let s = &schema(&model(&[(0, 2), (1, 2), (4, 2)]), &model(&[(0, 2), (1, 3), (2, 2)]))[0];
    assert_eq!(kinds(&s.base), ["Direct", "Unsupported", "Gap", "Gap", "NeedsMapping"]);
    assert_eq!(kinds(&s.donor), ["Direct", "Unsupported", "NeedsMapping"]);
    assert_eq!(s.differences[&1], ["w: F32 [2] vs F32 [3]"]);
}

#[test]
fn only_direct_transfer_is_available_and_adapters_are_never_inferred() {
    assert_eq!(Transfer::ALL.iter().filter(|t| t.available()).collect::<Vec<_>>(), [&Transfer::Direct]);
    for status in [
        BlockStatus::Compatible,
        BlockStatus::Different(vec!["hidden size".into()]),
        BlockStatus::OnlyInBase,
        BlockStatus::OnlyInDonor,
    ] {
        assert_ne!(Transfer::of(&status), Transfer::NeedsAdapter, "{status:?}");
    }
}

#[test]
fn todays_links_are_same_index_but_a_link_can_join_any_pair() {
    let blocks = [BlockId { stack: "model.layers".into(), index: 15 }];
    assert_eq!(same_index_links(&blocks), [Link { stack: "model.layers".into(), donor: 15, base: 15 }]);
    let cross = Link { stack: "model.layers".into(), donor: 3, base: 9 };
    assert_eq!(place(&cross, 10), Placement::Same { line: 0, donor_col: 3, base_col: 9 });
    assert_eq!(place(&cross, 5), Placement::Split { donor_line: 0, base_line: 1 });
    assert_eq!(
        place(&Link { donor: 15, base: 15, ..cross }, 8),
        Placement::Same { line: 1, donor_col: 7, base_col: 7 }
    );
}

#[test]
fn wrapped_lines_fit_640_1280_and_1920_pixels() {
    let label = 120.0;
    for width in [640.0_f32, 1280.0, 1920.0] {
        // The page keeps margins; the schema gets what is left.
        let inner = width - 64.0;
        let cols = columns(inner, label);
        assert!(label + cols as f32 * (CELL + GAP) - GAP <= inner, "{width}: {cols} columns overflow");
        assert!(label + (cols + 1) as f32 * (CELL + GAP) - GAP > inner, "{width}: room for one more column");
        for n in [0, 1, 30, 100] {
            let l = lines(n, cols);
            assert_eq!(l.iter().map(|r| r.len()).sum::<usize>(), n, "{width}px, {n} blocks");
            assert!(l.iter().all(|r| r.len() <= cols));
        }
    }
    assert_eq!(columns(10.0, 120.0), 1, "even a tiny window shows one column");
}
