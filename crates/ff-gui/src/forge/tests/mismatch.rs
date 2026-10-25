//! Tests for module `forge::mismatch`: each kind of difference becomes its own reason.

use ff_core::{DType, Model, TensorInfo};
use ff_surgery::compare::{self, TokenizerDiff};

use super::*;
use crate::i18n::Lang;

fn model(tensors: &[(&str, usize)]) -> Model {
    let mut m = Model::default();
    for &(name, width) in tensors {
        m.insert(TensorInfo { name: name.into(), dtype: DType::F32, shape: vec![width] });
    }
    m
}

fn report(base: &Model, donor: &Model, tokenizer: TokenizerMatch) -> Report {
    Report {
        a: "base".into(),
        b: "donor".into(),
        tensors: compare::tensors(base, donor),
        stacks: compare::stacks(base, donor),
        tokenizer: TokenizerDiff { status: tokenizer, vocab_a: None, vocab_b: None },
    }
}

#[test]
fn identical_structure_has_nothing_to_report() {
    let m = model(&[("model.layers.0.w", 2), ("model.layers.1.w", 2), ("model.norm.weight", 2)]);
    assert_eq!(mismatches(&report(&m, &m, TokenizerMatch::Identical)), []);
}

#[test]
fn hidden_size_block_count_and_tokenizer_are_separate_reasons() {
    let base = model(&[
        ("model.embed_tokens.weight", 8),
        ("model.layers.0.w", 8),
        ("model.layers.1.w", 8),
        ("model.layers.2.w", 8),
    ]);
    let donor = model(&[("model.embed_tokens.weight", 6), ("model.layers.0.w", 6), ("model.layers.1.w", 6)]);
    let found = mismatches(&report(&base, &donor, TokenizerMatch::DifferentVocab));
    assert_eq!(
        found,
        [
            Mismatch::Tokenizer(TokenizerMatch::DifferentVocab),
            Mismatch::BlockCount { stack: "model.layers".into(), base: 3, donor: 2 },
            Mismatch::BlockShapes {
                stack: "model.layers".into(),
                count: 2,
                index: 0,
                example: "w: F32 [8] vs F32 [6]".into()
            },
            Mismatch::OuterShapes { count: 1, example: "model.embed_tokens.weight".into() },
        ]
    );
}

#[test]
fn an_unrecognized_architecture_is_named_as_such() {
    let base = model(&[("model.layers.0.w", 2)]);
    let flat = model(&[("weights", 2)]);
    let found = mismatches(&report(&base, &flat, TokenizerMatch::Identical));
    assert!(found.contains(&Mismatch::NoBlocks), "{found:?}");
    assert!(found.contains(&Mismatch::StackOnlyIn { stack: "model.layers".into(), side: Side::Base }));
    assert!(found.contains(&Mismatch::OuterMissing { base: 0, donor: 1 }));
}

#[test]
fn every_reason_reads_as_a_sentence_in_every_language() {
    let all = [
        Mismatch::Tokenizer(TokenizerMatch::Missing),
        Mismatch::NoBlocks,
        Mismatch::StackOnlyIn { stack: "m.layers".into(), side: Side::Base },
        Mismatch::StackOnlyIn { stack: "m.layers".into(), side: Side::Donor },
        Mismatch::BlockCount { stack: "m.layers".into(), base: 30, donor: 24 },
        Mismatch::BlockShapes {
            stack: "m.layers".into(),
            count: 2,
            index: 0,
            example: "w: [8] vs [6]".into(),
        },
        Mismatch::OuterShapes { count: 1, example: "model.norm.weight".into() },
        Mismatch::OuterMissing { base: 1, donor: 2 },
    ];
    for lang in Lang::ALL {
        let t = I18n::new(lang);
        for m in &all {
            let text = m.text(&t);
            assert!(!text.starts_with("forge-") && text.len() > 10, "{}: {m:?} -> {text}", lang.code());
        }
    }
}
