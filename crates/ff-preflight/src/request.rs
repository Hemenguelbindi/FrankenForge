//! What an operation asks for. Sizes come from the selected tensors of a real checkpoint header,
//! never from a model's name.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ff_core::Model;

use crate::attention::Attention;
use crate::known::Known;

/// A group of tensors, e.g. `model.layers` or `model.vision_tower`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub name: String,
    pub params: u64,
    /// Bytes as stored in the checkpoint.
    pub stored_bytes: u64,
}

/// Groups tensors by their first two name segments (`model.layers.3.mlp.w` -> `model.layers`);
/// a name with fewer segments is its own group.
pub fn components(model: &Model) -> Vec<Component> {
    let mut groups: BTreeMap<String, Component> = BTreeMap::new();
    for t in model.tensors.values() {
        let parts: Vec<&str> = t.name.splitn(3, '.').collect();
        let key = if parts.len() >= 3 { format!("{}.{}", parts[0], parts[1]) } else { t.name.clone() };
        let c = groups.entry(key.clone()).or_insert(Component { name: key, params: 0, stored_bytes: 0 });
        c.params += t.numel() as u64;
        c.stored_bytes += t.byte_len() as u64;
    }
    groups.into_values().collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Bf16,
    /// Only meaningful if the chosen runtime supports it; the estimate says so.
    Int8,
}

impl Precision {
    /// Bytes per parameter as a range (int8 adds per-group scales: up to 1/8 extra).
    pub fn bytes_per_param(self) -> (f64, f64) {
        match self {
            Precision::F32 => (4.0, 4.0),
            Precision::Bf16 => (2.0, 2.0),
            Precision::Int8 => (1.0, 1.125),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Gpu,
    /// Weights and compute in system RAM on the CPU: avoids VRAM, costs speed.
    CpuOffload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildRequest {
    /// Weights plus companion files of the result.
    pub output_bytes: u64,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRequest {
    pub components: Vec<Component>,
    pub precision: Precision,
    pub attention: Known<Attention>,
    pub context: u64,
    pub batch: u64,
    pub placement: Placement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRequest {
    /// Full fine-tuning: every selected component is trainable.
    pub trainable: Vec<Component>,
    pub frozen: Vec<Component>,
    pub attention: Known<Attention>,
    pub context: u64,
    pub batch: u64,
    pub placement: Placement,
    pub checkpoints_kept: u64,
    pub companion_bytes: u64,
    /// Where checkpoints are written; `None` leaves the disk check Unknown.
    pub out_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceRequest {
    Build(BuildRequest),
    Inference(InferenceRequest),
    Training(TrainingRequest),
}

pub fn params(components: &[Component]) -> u64 {
    components.iter().map(|c| c.params).sum()
}
