//! Tests for module `preflight::state`: the request follows the chosen components and scenario,
//! and every report element has its own message key.

use ff_preflight::estimate;

use super::*;

fn opened() -> Opened {
    let c = |name: &str, params| Component { name: name.into(), params, stored_bytes: params * 2 };
    Opened {
        path: "m".into(),
        components: vec![
            c("model.language_model", 1_000),
            c("model.vision_tower", 300),
            c("model.audio_tower", 200),
        ],
        attention: Known::unknown("no config.json"),
        companion_bytes: 50,
    }
}

#[test]
fn unticked_components_are_left_out_of_the_estimate() {
    let mut s = Settings::default();
    let weights = |s: &Settings| estimate(&s.request(&opened())).items[0].size.get().unwrap().high;
    assert_eq!(weights(&s), (1_000 + 300 + 200) * 2);
    s.excluded.insert("model.vision_tower".into());
    s.excluded.insert("model.audio_tower".into());
    assert_eq!(weights(&s), 1_000 * 2, "text-only teacher");
}

#[test]
fn training_is_full_fine_tuning_with_its_own_disk_target() {
    let mut s = Settings { scenario: Scenario::Training, precision: Precision::Int8, ..Settings::default() };
    s.checkpoint_dir = Some("ckpt".into());
    let ResourceRequest::Training(t) = s.request(&opened()) else { panic!("training request expected") };
    assert_eq!(t.trainable.len(), 3);
    assert!(t.frozen.is_empty());
    assert_eq!(t.out_dir, Some("ckpt".into()));
    assert_eq!(t.companion_bytes, 50);
    // Precision is an inference choice; training stays f32 whatever was picked.
    let e = estimate(&ResourceRequest::Training(t));
    assert!(e.assumptions.contains(&Assumption::FullFineTuneF32AdamW));
}

#[test]
fn an_unknown_model_shape_is_passed_through_not_guessed() {
    let ResourceRequest::Inference(i) = Settings::default().request(&opened()) else { panic!() };
    assert!(matches!(i.attention, Known::Unknown(_)));
}

#[test]
fn every_report_element_has_a_distinct_key() {
    let mut keys: Vec<&str> = STATUSES.iter().flat_map(|s| <[&str; 2]>::from(status_keys(*s))).collect();
    keys.extend(RESOURCES.map(resource_key));
    keys.extend(PARTS.map(part_key));
    keys.extend(ASSUMPTIONS.map(assumption_key));
    keys.extend(ADVICE.map(advice_key));
    let mut unique = keys.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(keys.len(), unique.len());
}
