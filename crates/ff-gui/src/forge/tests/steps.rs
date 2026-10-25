//! Tests for module `forge::steps`: the strip follows the real state, never ahead of it.

use ff_surgery::compare::{Report, StackDiff, TensorDiff, TokenizerDiff, TokenizerMatch};

use super::*;
use crate::failure::Failure;
use crate::forge::analysis::{Analysis, BlockId};
use crate::forge::build::Built;
use crate::forge::readiness::{Checks, Inputs};
use crate::forge::state::Preview;

fn fine(i: &Inputs) -> Checks {
    Checks { same_model: false, output: i.out_dir.as_ref().map(|_| Ok(())), disk: None }
}

fn analysis(tokenizer: TokenizerMatch) -> Analysis {
    Analysis::new(Report {
        a: "a".into(),
        b: "b".into(),
        tensors: TensorDiff::default(),
        stacks: vec![StackDiff {
            stack: "m".into(),
            blocks_a: 1,
            blocks_b: 1,
            compatible: vec![0],
            incompatible: vec![],
            only_a: vec![],
            only_b: vec![],
        }],
        tokenizer: TokenizerDiff { status: tokenizer, vocab_a: None, vocab_b: None },
    })
}

fn now(f: &mut Forge) -> [Mark; 4] {
    let problems = f.problems_with(fine);
    marks(f, &problems)
}

use Mark::{Attention, Current, Done, Waiting};

#[test]
fn the_strip_advances_with_each_real_choice() {
    let mut f = Forge::default();
    assert_eq!(now(&mut f), [Current, Waiting, Waiting, Waiting]);
    f.set_base("b".into());
    assert_eq!(now(&mut f), [Done, Current, Waiting, Waiting]);
    f.set_donor("d".into());
    assert_eq!(now(&mut f), [Done, Done, Waiting, Waiting], "analysis still running");
    f.apply_analysis(f.pair().unwrap(), Ok(analysis(TokenizerMatch::Identical)));
    assert_eq!(now(&mut f), [Done, Done, Current, Waiting]);
    f.toggle(&BlockId { stack: "m".into(), index: 0 });
    assert_eq!(now(&mut f), [Done, Done, Current, Waiting], "preview not back yet");
    let sel = f.selection().unwrap();
    f.apply_preview(sel, Ok(Preview { from_base: 0, from_donor: 1, notes: vec![], output_bytes: 0 }));
    assert_eq!(now(&mut f), [Done, Done, Done, Waiting], "no output chosen");
    f.set_out_dir("out".into());
    f.set_out_name("x".into());
    assert_eq!(now(&mut f), [Done, Done, Done, Current]);

    let req = f.begin_build_with(fine).unwrap();
    assert_eq!(now(&mut f)[3], Current);
    let built = Built {
        out: req.out.clone(),
        recipe: "r".into(),
        lock: "l".into(),
        log: None,
        tensors: 1,
        params: 1,
        planned: req.preview.clone(),
        warnings: vec![],
    };
    f.finish_build(req, Ok(Ok(built)));
    assert_eq!(now(&mut f)[3], Done);
}

#[test]
fn problems_are_marked_as_needing_attention() {
    let mut f = Forge::default();
    f.set_base("b".into());
    f.set_donor("d".into());
    f.apply_analysis(f.pair().unwrap(), Ok(analysis(TokenizerMatch::DifferentVocab)));
    assert_eq!(now(&mut f)[2], Attention);

    let mut f = Forge::default();
    f.set_base("b".into());
    f.set_donor("d".into());
    f.apply_analysis(f.pair().unwrap(), Err(Failure::crashed()));
    assert_eq!(now(&mut f)[2], Attention);
}

#[test]
fn a_failed_build_is_attention_not_done() {
    let mut f = Forge::default();
    f.set_base("b".into());
    f.set_donor("d".into());
    f.apply_analysis(f.pair().unwrap(), Ok(analysis(TokenizerMatch::Identical)));
    f.toggle(&BlockId { stack: "m".into(), index: 0 });
    let sel = f.selection().unwrap();
    f.apply_preview(sel, Ok(Preview { from_base: 0, from_donor: 1, notes: vec![], output_bytes: 0 }));
    f.set_out_dir("out".into());
    let req = f.begin_build_with(fine).unwrap();
    f.finish_build(req, Err(Failure::crashed()));
    assert_eq!(now(&mut f)[3], Attention);
}

#[test]
fn every_mark_has_its_own_words() {
    let keys = [Done, Current, Waiting, Attention].map(Mark::message_key);
    assert_eq!(keys, Mark::KEYS);
}
