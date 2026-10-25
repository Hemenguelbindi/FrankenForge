//! Tests for module `learn`: explanation selection and the route's fixed content keys.

use super::*;

#[test]
fn explanations_open_one_at_a_time_and_toggle_closed() {
    let mut state = LearnState::default();
    assert_eq!(state.open(), None);

    state.toggle(Explanation::Attention);
    assert!(state.is_open(Explanation::Attention));
    assert!(!state.is_open(Explanation::What));

    state.toggle(Explanation::Tensor);
    assert!(state.is_open(Explanation::Tensor));
    assert!(!state.is_open(Explanation::Attention), "choosing another closes the previous one");

    state.toggle(Explanation::Tensor);
    assert_eq!(state.open(), None, "clicking the open explanation closes it");
}

#[test]
fn every_explanation_has_a_distinct_title_and_body() {
    let mut keys = Vec::new();
    for e in Explanation::ALL {
        keys.push(e.title_key());
        keys.push(e.body_key());
    }
    let mut unique = keys.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(keys.len(), unique.len(), "an explanation reuses another explanation's key");
    assert!(keys.iter().all(|k| k.starts_with("explain-")));
}

#[test]
fn the_route_has_five_numbered_steps() {
    assert_eq!(STEPS.len(), 5);
    let mut keys = Vec::new();
    for (title, body) in STEPS {
        keys.push(title);
        keys.push(body);
    }
    let mut unique = keys.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(keys.len(), unique.len());
    assert!(keys.iter().all(|k| k.starts_with("learn-step-")));
}
