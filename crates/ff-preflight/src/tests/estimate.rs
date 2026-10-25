//! Tests for module `estimate`: each part is counted on its own resource, with its assumptions.

use std::path::PathBuf;

use super::*;
use crate::request::Component;

fn layers(params: u64) -> Vec<Component> {
    vec![Component { name: "model.layers".into(), params, stored_bytes: params * 2 }]
}

#[test]
fn a_build_needs_one_partial_copy_and_a_reserve_on_disk() {
    let e = estimate(&ResourceRequest::Build(BuildRequest {
        output_bytes: 100 * GIB,
        out_dir: PathBuf::from("o"),
    }));
    assert_eq!(e.resources(), [Resource::Disk]);
    assert_eq!(e.need(Resource::Disk).span, Span::exact(102 * GIB));
    assert!(e.assumptions.contains(&Assumption::RenameOnSameVolume));
}

#[test]
fn inference_precision_changes_weights_and_int8_is_flagged() {
    let req = |precision| {
        ResourceRequest::Inference(InferenceRequest {
            components: layers(1_000),
            precision,
            attention: Known::unknown("not needed here"),
            context: 1,
            batch: 1,
            placement: Placement::Gpu,
        })
    };
    let weights = |p| estimate(&req(p)).items[0].size.get().copied().unwrap();
    assert_eq!(weights(Precision::F32), Span::exact(4_000));
    assert_eq!(weights(Precision::Bf16), Span::exact(2_000));
    assert_eq!(weights(Precision::Int8), Span::between(1_000, 1_125));
    assert!(estimate(&req(Precision::Int8)).assumptions.contains(&Assumption::Int8NeedsRuntimeSupport));
    assert!(!estimate(&req(Precision::Bf16)).assumptions.contains(&Assumption::Int8NeedsRuntimeSupport));
}

#[test]
fn every_inference_part_is_listed_even_when_zero() {
    let e = estimate(&ResourceRequest::Inference(InferenceRequest {
        components: layers(10),
        precision: Precision::Bf16,
        attention: Known::unknown("x"),
        context: 1,
        batch: 1,
        placement: Placement::Gpu,
    }));
    let parts: Vec<Part> = e.items.iter().map(|i| i.part).collect();
    assert_eq!(
        parts,
        [Part::Weights, Part::KvCache, Part::Activations, Part::Adapters, Part::RuntimeOverhead]
    );
}
