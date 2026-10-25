//! Preflight panel choices and the request they describe, plus the message keys for every part
//! of a report. No egui and no I/O here; the hardware snapshot and the model are read elsewhere.

use std::collections::BTreeSet;
use std::path::PathBuf;

use ff_preflight::attention::Attention;
use ff_preflight::{
    Advice, Assumption, Component, InferenceRequest, Known, Part, Placement, Precision, PreflightStatus,
    Resource, ResourceRequest, TrainingRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    Inference,
    Training,
}

/// A model folder read from its headers and config.json only.
#[derive(Debug, Clone)]
pub struct Opened {
    pub path: PathBuf,
    pub components: Vec<Component>,
    pub attention: Known<Attention>,
    /// Non-weight files a training checkpoint copies next to the weights.
    pub companion_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub scenario: Scenario,
    pub precision: Precision,
    pub placement: Placement,
    pub context: u64,
    pub batch: u64,
    pub checkpoints: u64,
    pub checkpoint_dir: Option<PathBuf>,
    /// Components left out; everything else is loaded.
    pub excluded: BTreeSet<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            scenario: Scenario::Inference,
            precision: Precision::Bf16,
            placement: Placement::Gpu,
            context: 2048,
            batch: 1,
            checkpoints: 1,
            checkpoint_dir: None,
            excluded: BTreeSet::new(),
        }
    }
}

impl Settings {
    pub fn chosen(&self, m: &Opened) -> Vec<Component> {
        m.components.iter().filter(|c| !self.excluded.contains(&c.name)).cloned().collect()
    }

    /// Training is full fine-tuning in f32 (the confirmed backend), so precision applies to
    /// inference only.
    pub fn request(&self, m: &Opened) -> ResourceRequest {
        let chosen = self.chosen(m);
        match self.scenario {
            Scenario::Inference => ResourceRequest::Inference(InferenceRequest {
                components: chosen,
                precision: self.precision,
                attention: m.attention.clone(),
                context: self.context,
                batch: self.batch,
                placement: self.placement,
            }),
            Scenario::Training => ResourceRequest::Training(TrainingRequest {
                trainable: chosen,
                frozen: Vec::new(),
                attention: m.attention.clone(),
                context: self.context,
                batch: self.batch,
                placement: self.placement,
                checkpoints_kept: self.checkpoints,
                companion_bytes: m.companion_bytes,
                out_dir: self.checkpoint_dir.clone(),
            }),
        }
    }
}

#[cfg(test)]
pub const STATUSES: [PreflightStatus; 4] =
    [PreflightStatus::Fits, PreflightStatus::Tight, PreflightStatus::Insufficient, PreflightStatus::Unknown];

/// (word, explanation)
pub fn status_keys(s: PreflightStatus) -> (&'static str, &'static str) {
    match s {
        PreflightStatus::Fits => ("pf-status-fits", "pf-status-fits-desc"),
        PreflightStatus::Tight => ("pf-status-tight", "pf-status-tight-desc"),
        PreflightStatus::Insufficient => ("pf-status-insufficient", "pf-status-insufficient-desc"),
        PreflightStatus::Unknown => ("pf-status-unknown", "pf-status-unknown-desc"),
    }
}

#[cfg(test)]
pub const RESOURCES: [Resource; 3] = [Resource::Vram, Resource::Ram, Resource::Disk];

pub fn resource_key(r: Resource) -> &'static str {
    match r {
        Resource::Vram => "pf-res-vram",
        Resource::Ram => "pf-res-ram",
        Resource::Disk => "pf-res-disk",
    }
}

#[cfg(test)]
pub const PARTS: [Part; 13] = [
    Part::PartialCopy,
    Part::DiskReserve,
    Part::Weights,
    Part::FrozenWeights,
    Part::TrainableWeights,
    Part::Gradients,
    Part::OptimizerStates,
    Part::KvCache,
    Part::Activations,
    Part::Logits,
    Part::Adapters,
    Part::RuntimeOverhead,
    Part::Checkpoints,
];

pub fn part_key(p: Part) -> &'static str {
    match p {
        Part::PartialCopy => "pf-part-partial",
        Part::DiskReserve => "pf-part-reserve",
        Part::Weights => "pf-part-weights",
        Part::FrozenWeights => "pf-part-frozen",
        Part::TrainableWeights => "pf-part-trainable",
        Part::Gradients => "pf-part-gradients",
        Part::OptimizerStates => "pf-part-optimizer",
        Part::KvCache => "pf-part-kv",
        Part::Activations => "pf-part-activations",
        Part::Logits => "pf-part-logits",
        Part::Adapters => "pf-part-adapters",
        Part::RuntimeOverhead => "pf-part-runtime",
        Part::Checkpoints => "pf-part-checkpoints",
    }
}

#[cfg(test)]
pub const ASSUMPTIONS: [Assumption; 11] = [
    Assumption::RenameOnSameVolume,
    Assumption::DiskReserve,
    Assumption::Int8NeedsRuntimeSupport,
    Assumption::KvFromConfig,
    Assumption::ActivationsRough,
    Assumption::RuntimeOverheadTypical,
    Assumption::CpuOffloadIsSlow,
    Assumption::NoAdapters,
    Assumption::FullFineTuneF32AdamW,
    Assumption::TrainingOnGpuUnverified,
    Assumption::SingleGpu,
];

pub fn assumption_key(a: Assumption) -> &'static str {
    match a {
        Assumption::RenameOnSameVolume => "pf-assume-rename",
        Assumption::DiskReserve => "pf-assume-reserve",
        Assumption::Int8NeedsRuntimeSupport => "pf-assume-int8",
        Assumption::KvFromConfig => "pf-assume-kv",
        Assumption::ActivationsRough => "pf-assume-activations",
        Assumption::RuntimeOverheadTypical => "pf-assume-runtime",
        Assumption::CpuOffloadIsSlow => "pf-assume-offload",
        Assumption::NoAdapters => "pf-assume-no-adapters",
        Assumption::FullFineTuneF32AdamW => "pf-assume-full-ft",
        Assumption::TrainingOnGpuUnverified => "pf-assume-gpu-training",
        Assumption::SingleGpu => "pf-assume-single-gpu",
    }
}

#[cfg(test)]
pub const ADVICE: [Advice; 6] = [
    Advice::ReduceBatch,
    Advice::ReduceContext,
    Advice::FewerComponents,
    Advice::TryCpuOffload,
    Advice::FreeDisk,
    Advice::Reprobe,
];

pub fn advice_key(a: Advice) -> &'static str {
    match a {
        Advice::ReduceBatch => "pf-advice-batch",
        Advice::ReduceContext => "pf-advice-context",
        Advice::FewerComponents => "pf-advice-components",
        Advice::TryCpuOffload => "pf-advice-offload",
        Advice::FreeDisk => "pf-advice-disk",
        Advice::Reprobe => "pf-advice-reprobe",
    }
}

#[cfg(test)]
#[path = "tests/state.rs"]
mod tests;
