//! Tests for module `preflight`: real checkpoints read through `open_model` (headers and
//! config.json only), estimated against this machine. Ignored; paths come from the environment.
//!
//! ```powershell
//! $env:FF_SMOLLM2_RECIPIENT = '<models>\SmolLM2-135M'
//! $env:FF_GEMMA_E2B = '<models>\gemma-4-E2B-it'
//! cargo test -p ff-gui preflight::tests -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use ff_preflight::{PreflightReport, ResourceRequest, preflight};

use super::*;

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("set {var} to a model directory")))
}

fn print(label: &str, r: &PreflightReport) {
    println!("--- {label}: {:?}", r.status);
    for i in &r.estimate.items {
        println!("    {:?} on {:?}: {:?}", i.part, i.resource, i.size);
    }
    for c in &r.checks {
        println!(
            "    check {:?}: need {:?} (+unknown {:?}) available {:?} -> {:?}",
            c.resource, c.need.span, c.need.unknown, c.available, c.status
        );
    }
    println!("    advice {:?}", r.advice);
}

#[test]
#[ignore = "needs FF_SMOLLM2_RECIPIENT and FF_GEMMA_E2B"]
fn real_models_against_this_machine() {
    let snap = probe::probe();
    let smol = open_model(&env_path("FF_SMOLLM2_RECIPIENT")).unwrap();
    let gemma = open_model(&env_path("FF_GEMMA_E2B")).unwrap();
    for (name, m) in [("SmolLM2-135M", &smol), ("gemma-4-E2B-it", &gemma)] {
        println!("== {name}: shape {:?}", m.attention.get().map(|a| (a.layers, a.hidden, a.caches.len())));
        for c in &m.components {
            println!("    component {} params {} stored {}", c.name, c.params, c.stored_bytes);
        }
    }
    assert!(smol.attention.get().is_some() && gemma.attention.get().is_some());

    let mut s = Settings::default();
    print("SmolLM2 BF16 GPU ctx 2048", &preflight(&s.request(&smol), &snap));
    let train = Settings {
        scenario: Scenario::Training,
        context: 512,
        batch: 1,
        placement: Placement::CpuOffload,
        ..Settings::default()
    };
    print("SmolLM2 full FT f32 CPU ctx 512 batch 1", &preflight(&train.request(&smol), &snap));

    // Text-only teacher: leave out every component whose name mentions vision or audio.
    for c in &gemma.components {
        if c.name.contains("vision") || c.name.contains("audio") {
            s.excluded.insert(c.name.clone());
        }
    }
    println!("Gemma text-only excludes {:?}", s.excluded);
    for (label, precision, placement) in [
        ("Gemma text BF16 GPU ctx 2048", Precision::Bf16, Placement::Gpu),
        ("Gemma text 8-bit GPU ctx 2048 (runtime support unverified)", Precision::Int8, Placement::Gpu),
        ("Gemma text BF16 CPU offload ctx 2048", Precision::Bf16, Placement::CpuOffload),
    ] {
        let set = Settings { precision, placement, ..s.clone() };
        let ResourceRequest::Inference(_) = set.request(&gemma) else { panic!() };
        print(label, &preflight(&set.request(&gemma), &snap));
    }
}
