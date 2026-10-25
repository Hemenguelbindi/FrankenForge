//! Tests for module `inspector`: a finished analysis belongs to exactly one tensor.

use super::*;
use crate::analysis::{BINS, Finite};
use crate::i18n::Lang;

fn selection(name: &str) -> Selection {
    let info = TensorInfo { name: name.into(), dtype: ff_core::DType::BF16, shape: vec![2, 2] };
    Selection { doc: describe(&info, None), info, component: None, block: None, name: name.into() }
}

fn state(name: &str) -> InspectorState {
    InspectorState { selection: Some(selection(name)), ..Default::default() }
}

fn ordinary_profile() -> Profile {
    let finite = Finite { count: 4, min: -1.0, max: 2.0, mean: 0.5, std: 1.0, histogram: vec![0; BINS] };
    Profile { count: 4, zeros: 1, nan: 0, pos_inf: 0, neg_inf: 0, finite: Some(finite) }
}

#[test]
fn modes_default_to_simple_and_round_trip() {
    assert_eq!(Mode::default(), Mode::Simple);
    assert_eq!(InspectorState::default().mode(), Mode::Simple);
    for mode in [Mode::Simple, Mode::Expert] {
        assert_eq!(Mode::from_code(mode.code()), Some(mode));
        assert_eq!(Mode::from_code(&mode.code().to_uppercase()), Some(mode), "saved codes ignore case");
    }
    // A corrupt or older saved value falls back to the default.
    assert_eq!(Mode::from_code("verbose"), None);
    assert_eq!(Mode::from_code(""), None);
}

#[test]
fn reset_keeps_the_chosen_mode() {
    let mut state = InspectorState::default();
    state.set_mode(Mode::Expert);
    state.reset();
    assert_eq!(state.mode(), Mode::Expert);
}

#[test]
fn stop_analysis_keeps_the_selection_and_abandons_the_pending_read() {
    let mut state = state("model.layers.0.self_attn.q_proj.weight");
    state.loader.start(
        state.selected().unwrap().to_owned(),
        |_| Err(crate::failure::Failure::crashed()),
        || {},
    );

    state.stop_analysis();
    assert_eq!(state.selected(), Some("model.layers.0.self_attn.q_proj.weight"));
    assert!(state.loader.pending().is_none(), "the running analysis was abandoned");
    // Without a pending request it must stay harmless.
    state.stop_analysis();
    assert_eq!(state.selected(), Some("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
fn ordinary_profile_conclusion_is_deterministic_and_plain() {
    let t = I18n::new(Lang::En);
    let p = ordinary_profile();
    let text = conclusion(&t, &p).expect("a profile conclusion");
    assert_eq!(
        text,
        "Values range from -1.000000 to 2.000000, with mean 0.500000 and standard deviation \
         1.000000. These numbers describe only the scale and spread of the stored values."
    );
    assert_eq!(text, conclusion(&t, &p).expect("the same profile gives the same text"));
    for forbidden in ["healthy", "normal", "model", "know"] {
        assert!(!text.to_lowercase().contains(forbidden), "{forbidden} in {text}");
    }
}

#[test]
fn nonfinite_and_all_zero_conclusions_are_explicit() {
    let t = I18n::new(Lang::En);
    let nonfinite = Profile { count: 3, zeros: 0, nan: 2, pos_inf: 1, neg_inf: 0, finite: None };
    assert_eq!(
        conclusion(&t, &nonfinite).expect("a non-finite conclusion"),
        "Non-finite values: 2 NaN, 1 positive infinity, 0 negative infinity. This usually means \
         a damaged file or a conversion error."
    );

    let finite = Finite { count: 5, min: 0.0, max: 0.0, mean: 0.0, std: 0.0, histogram: vec![0; BINS] };
    let zeros = Profile { count: 5, zeros: 5, nan: 0, pos_inf: 0, neg_inf: 0, finite: Some(finite) };
    assert_eq!(conclusion(&t, &zeros).expect("an all-zero conclusion"), "All 5 elements are exactly zero.");

    // Non-finite values win over the all-zero rule: the order is fixed.
    let both = Profile { nan: 1, ..zeros };
    assert!(conclusion(&t, &both).expect("a conclusion").starts_with("Non-finite values"));
}

#[test]
fn conclusions_use_the_locale_decimal_separator() {
    let ru = I18n::new(Lang::Ru);
    let text = conclusion(&ru, &ordinary_profile()).expect("a profile conclusion");
    assert!(text.contains("0,500000"), "decimal comma in {text}");
    assert!(!text.contains("0.500000"), "no decimal point in {text}");
}

#[test]
fn a_result_for_another_tensor_is_dropped() {
    let mut state = state("model.layers.0.self_attn.q_proj.weight");
    state.accept(("model.layers.1.self_attn.q_proj.weight".into(), Ok(Analysis::Empty)));
    assert_eq!(state.result, None, "a stale result must not be shown");

    state.accept(("model.layers.0.self_attn.q_proj.weight".into(), Ok(Analysis::Empty)));
    assert!(state.result.is_some(), "the selected tensor's own result is kept");
}

#[test]
fn reset_drops_the_selection_and_its_result() {
    let mut state = state("a.weight");
    state.accept(("a.weight".into(), Ok(Analysis::Empty)));
    state.reset();
    assert_eq!(state.selected(), None);
    assert_eq!(state.result, None);
}
