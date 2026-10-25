//! Tests for module `report` on synthetic snapshots: fits, tight, short disk, short VRAM,
//! unknown VRAM, CPU offload as its own scenario, and sizes taken from the chosen components.

use std::path::PathBuf;

use ff_core::{DType, Model, TensorInfo};

use super::*;
use crate::attention::Attention;
use crate::estimate::{Assumption, Part};
use crate::known::{GIB, MIB};
use crate::request::{BuildRequest, Component, InferenceRequest, Precision, TrainingRequest, components};
use crate::snapshot::{DiskInfo, GpuInfo};

fn snap(vram_free: Option<u64>, ram: u64, disk: u64) -> HardwareSnapshot {
    HardwareSnapshot {
        taken_at_unix: 1_700_000_000,
        cpu: Known::Known("test cpu".into()),
        logical_cores: Known::Known(8),
        ram_total: Known::Known(ram * 2),
        ram_available: Known::Known(ram),
        ram_source: "test".into(),
        gpus: match vram_free {
            Some(free) => {
                Known::Known(vec![GpuInfo { name: "test gpu".into(), vram_total: 12 * GIB, vram_free: free }])
            }
            None => Known::unknown("nvidia-smi not found; this does not prove there is no NVIDIA GPU"),
        },
        gpu_source: "test".into(),
        disks: Known::Known(vec![DiskInfo {
            mount: out_dir().parent().unwrap().to_owned(),
            total: 4 * disk,
            available: disk,
        }]),
        disk_source: "test".into(),
    }
}

fn out_dir() -> PathBuf {
    std::env::temp_dir().join("ff-preflight-out")
}

fn attention() -> Attention {
    let config = serde_json::json!({
        "hidden_size": 576, "intermediate_size": 1536, "num_attention_heads": 9,
        "num_key_value_heads": 3, "num_hidden_layers": 30, "vocab_size": 49152
    });
    Attention::from_config(&config).get().cloned().unwrap()
}

fn component(name: &str, params: u64) -> Component {
    Component { name: name.into(), params, stored_bytes: params * 2 }
}

fn inference(params: u64, placement: Placement) -> ResourceRequest {
    ResourceRequest::Inference(InferenceRequest {
        components: vec![component("model.layers", params)],
        precision: Precision::Bf16,
        attention: Known::Known(attention()),
        context: 2048,
        batch: 1,
        placement,
    })
}

fn status_of(r: &PreflightReport, resource: Resource) -> PreflightStatus {
    r.checks.iter().find(|c| c.resource == resource).unwrap().status
}

#[test]
fn enough_of_everything_fits() {
    let r = preflight(&inference(135_000_000, Placement::Gpu), &snap(Some(10 * GIB), 32 * GIB, 100 * GIB));
    assert_eq!(r.status, PreflightStatus::Fits, "{:?}", r.checks);
    assert!(r.advice.is_empty());
}

#[test]
fn close_to_the_limit_is_tight_not_fits() {
    let need = estimate(&inference(135_000_000, Placement::Gpu)).need(Resource::Vram);
    // Between the high end of the need and the high end plus the 10% margin.
    let free = need.span.high + need.span.high / 20;
    let r = preflight(&inference(135_000_000, Placement::Gpu), &snap(Some(free), 32 * GIB, 100 * GIB));
    assert_eq!(status_of(&r, Resource::Vram), PreflightStatus::Tight);
    assert!(r.advice.contains(&Advice::ReduceContext) && r.advice.contains(&Advice::TryCpuOffload));
}

#[test]
fn a_disk_too_small_for_the_output_is_insufficient() {
    let req = ResourceRequest::Build(BuildRequest { output_bytes: 5 * GIB, out_dir: out_dir() });
    let r = preflight(&req, &snap(Some(10 * GIB), 32 * GIB, 3 * GIB));
    assert_eq!(r.status, PreflightStatus::Insufficient);
    assert_eq!(r.advice, [Advice::FreeDisk]);
    let ok = preflight(&req, &snap(Some(10 * GIB), 32 * GIB, 50 * GIB));
    assert_eq!(ok.status, PreflightStatus::Fits);
}

#[test]
fn not_enough_vram_is_insufficient() {
    // Gemma-E2B-sized weights (5.1 B parameters) in BF16 do not fit 4 GiB.
    let r = preflight(&inference(5_100_000_000, Placement::Gpu), &snap(Some(4 * GIB), 64 * GIB, 100 * GIB));
    assert_eq!(status_of(&r, Resource::Vram), PreflightStatus::Insufficient);
    assert!(r.advice.contains(&Advice::TryCpuOffload));
}

#[test]
fn unknown_vram_is_unknown_never_fits() {
    let r = preflight(&inference(1_000, Placement::Gpu), &snap(None, 64 * GIB, 100 * GIB));
    let vram = r.checks.iter().find(|c| c.resource == Resource::Vram).unwrap();
    assert_eq!(vram.status, PreflightStatus::Unknown);
    assert!(matches!(vram.available, Known::Unknown(ref why) if why.contains("does not prove")));
    assert_eq!(r.status, PreflightStatus::Unknown);
    assert!(r.advice.contains(&Advice::Reprobe));
}

#[test]
fn an_unsized_part_keeps_the_check_unknown() {
    let mut req = inference(1_000, Placement::Gpu);
    if let ResourceRequest::Inference(i) = &mut req {
        i.attention = Known::unknown("config.json has no `hidden_size`");
    }
    let r = preflight(&req, &snap(Some(10 * GIB), 64 * GIB, 100 * GIB));
    let vram = r.checks.iter().find(|c| c.resource == Resource::Vram).unwrap();
    assert_eq!(vram.need.unknown, [Part::KvCache, Part::Activations]);
    assert_eq!(vram.status, PreflightStatus::Unknown);
}

#[test]
fn cpu_offload_moves_the_need_to_ram_and_says_it_is_slow() {
    let req = inference(5_100_000_000, Placement::CpuOffload);
    let r = preflight(&req, &snap(Some(4 * GIB), 64 * GIB, 100 * GIB));
    assert!(r.checks.iter().all(|c| c.resource != Resource::Vram), "no VRAM need when offloaded");
    assert_eq!(status_of(&r, Resource::Ram), PreflightStatus::Fits);
    assert!(r.estimate.assumptions.contains(&Assumption::CpuOffloadIsSlow));
    assert!(!r.advice.contains(&Advice::TryCpuOffload));
    // Offload is not free: too little RAM is just as insufficient.
    let r = preflight(&req, &snap(Some(4 * GIB), 6 * GIB, 100 * GIB));
    assert_eq!(status_of(&r, Resource::Ram), PreflightStatus::Insufficient);
}

#[test]
fn sizes_follow_the_selected_tensors_not_the_model_name() {
    let model = |width: usize| {
        let mut m = Model::default();
        m.metadata.insert("name".into(), "same-name".into());
        for (name, rows) in [("model.layers.0.w", width), ("model.vision_tower.0.w", 1000)] {
            m.insert(TensorInfo { name: name.into(), dtype: DType::BF16, shape: vec![rows, 1000] });
        }
        m
    };
    let weights = |m: &Model, keep: &[&str]| {
        let chosen: Vec<Component> =
            components(m).into_iter().filter(|c| keep.contains(&c.name.as_str())).collect();
        let req = ResourceRequest::Inference(InferenceRequest {
            components: chosen,
            precision: Precision::Bf16,
            attention: Known::Known(attention()),
            context: 16,
            batch: 1,
            placement: Placement::Gpu,
        });
        estimate(&req).items[0].size.get().unwrap().high
    };
    let (small, large) = (model(100), model(400));
    assert_eq!(weights(&small, &["model.layers"]), 100 * 1000 * 2);
    assert_eq!(weights(&large, &["model.layers"]), 400 * 1000 * 2, "same name, other shapes");
    assert_eq!(weights(&small, &["model.layers", "model.vision_tower"]), 1100 * 1000 * 2);
}

#[test]
fn training_counts_states_separately_and_checks_the_checkpoint_disk() {
    let req = ResourceRequest::Training(TrainingRequest {
        trainable: vec![component("model.layers", 100 * 1_000_000)],
        frozen: vec![],
        attention: Known::Known(attention()),
        context: 256,
        batch: 2,
        placement: Placement::CpuOffload,
        checkpoints_kept: 1,
        companion_bytes: 3 * MIB,
        out_dir: Some(out_dir()),
    });
    // 806 MB of checkpoints plus the 1 GiB reserve do not fit 1 GiB.
    let r = preflight(&req, &snap(None, 64 * GIB, GIB));
    let size = |p: Part| r.estimate.items.iter().find(|i| i.part == p).unwrap().size.get().unwrap().high;
    assert_eq!(size(Part::TrainableWeights), 400 * 1_000_000);
    assert_eq!(size(Part::Gradients), 400 * 1_000_000);
    assert_eq!(size(Part::OptimizerStates), 800 * 1_000_000);
    assert_eq!(size(Part::Checkpoints), 2 * (400 * 1_000_000 + 3 * MIB));
    assert_eq!(status_of(&r, Resource::Disk), PreflightStatus::Insufficient);
    assert!(r.checks.iter().all(|c| c.resource != Resource::Vram), "CPU training needs no VRAM");
    assert!(r.advice.contains(&Advice::FreeDisk));
}
