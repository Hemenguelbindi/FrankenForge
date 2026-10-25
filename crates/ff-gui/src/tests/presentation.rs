//! Tests for module `presentation`: the saved navigation choice and its fallback.

use super::*;

#[test]
fn learn_is_the_default_and_the_tabs_are_ordered() {
    assert_eq!(Presentation::default(), Presentation::Learn);
    assert_eq!(
        Presentation::ALL,
        [Presentation::Learn, Presentation::Structure, Presentation::Technical],
        "the tour comes first for a new reader"
    );
}

#[test]
fn codes_round_trip_and_ignores_case_and_spaces() {
    for p in Presentation::ALL {
        assert_eq!(Presentation::from_code(p.code()), Some(p));
        assert_eq!(Presentation::from_code(&format!("  {}  ", p.code().to_uppercase())), Some(p));
    }
}

#[test]
fn a_missing_or_corrupt_saved_choice_falls_back_to_learn() {
    assert_eq!(initial_presentation(None), Presentation::Learn);
    assert_eq!(initial_presentation(Some("technical")), Presentation::Technical);
    assert_eq!(initial_presentation(Some("")), Presentation::Learn);
    assert_eq!(initial_presentation(Some("graph")), Presentation::Learn);
}

#[test]
fn every_tab_has_its_own_label_key() {
    let keys: Vec<&str> = Presentation::ALL.iter().map(|p| p.message_key()).collect();
    let mut unique = keys.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(keys.len(), unique.len());
    assert!(keys.iter().all(|k| k.starts_with("presentation-")));
}
