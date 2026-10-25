//! Tests for module `forge::analysis`: every block index of either model becomes a row, with the
//! status the comparison gave it.

use ff_core::{DType, Model, TensorInfo};
use ff_surgery::compare::{self, BlockDiff, StackDiff, TensorDiff, TokenizerDiff};

use super::*;

fn report(stacks: Vec<StackDiff>) -> Report {
    Report {
        a: "base".into(),
        b: "donor".into(),
        tensors: TensorDiff::default(),
        stacks,
        tokenizer: TokenizerDiff { status: TokenizerMatch::Identical, vocab_a: None, vocab_b: None },
    }
}

fn model(blocks: &[(usize, usize)]) -> Model {
    let mut m = Model::default();
    for &(index, width) in blocks {
        let name = format!("model.layers.{index}.mlp.weight");
        m.insert(TensorInfo { name, dtype: DType::F32, shape: vec![width] });
    }
    m
}

fn statuses(a: &Analysis) -> Vec<(usize, BlockStatus)> {
    a.rows.iter().map(|r| (r.id.index, r.status.clone())).collect()
}

#[test]
fn rows_cover_every_index_in_order_with_its_status() {
    let a = Analysis::new(report(vec![StackDiff {
        stack: "model.layers".into(),
        blocks_a: 3,
        blocks_b: 4,
        compatible: vec![1],
        incompatible: vec![BlockDiff { index: 5, differences: vec!["w: F32 [1] vs F32 [2]".into()] }],
        only_a: vec![0],
        only_b: vec![2, 3],
    }]));
    assert_eq!(
        statuses(&a),
        [
            (0, BlockStatus::OnlyInBase),
            (1, BlockStatus::Compatible),
            (2, BlockStatus::OnlyInDonor),
            (3, BlockStatus::OnlyInDonor),
            (5, BlockStatus::Different(vec!["w: F32 [1] vs F32 [2]".into()])),
        ]
    );
}

#[test]
fn a_base_with_more_blocks_shows_the_extra_ones_as_base_only() {
    let base = model(&[(0, 2), (1, 2), (2, 2), (3, 2)]);
    let donor = model(&[(0, 2), (1, 3)]);
    let a = Analysis::new(report(compare::stacks(&base, &donor)));
    assert_eq!(
        statuses(&a),
        [
            (0, BlockStatus::Compatible),
            (1, BlockStatus::Different(vec!["mlp.weight: F32 [2] vs F32 [3]".into()])),
            (2, BlockStatus::OnlyInBase),
            (3, BlockStatus::OnlyInBase),
        ]
    );
}

#[test]
fn a_donor_with_more_blocks_shows_the_extra_ones_as_donor_only() {
    let base = model(&[(0, 2)]);
    let donor = model(&[(0, 2), (1, 2), (2, 2)]);
    let a = Analysis::new(report(compare::stacks(&base, &donor)));
    assert_eq!(
        statuses(&a),
        [(0, BlockStatus::Compatible), (1, BlockStatus::OnlyInDonor), (2, BlockStatus::OnlyInDonor)]
    );
}

#[test]
fn only_compatible_rows_count_as_compatible() {
    let a = Analysis::new(report(compare::stacks(&model(&[(0, 2), (1, 2)]), &model(&[(0, 2)]))));
    let id = |index| BlockId { stack: "model.layers".into(), index };
    assert!(a.compatible(&id(0)));
    assert!(!a.compatible(&id(1)), "a block missing in the Donor is not selectable");
    assert!(!a.compatible(&id(7)), "an unknown index is not selectable");
}
