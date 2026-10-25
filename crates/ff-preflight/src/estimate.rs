//! What an operation needs, part by part. Each part has its own resource and a size range or
//! Unknown; assumptions that shape the numbers are listed next to them. There is no single
//! "parameters × magic factor": weights, caches, activations and states are counted separately.

use crate::attention::Attention;
use crate::known::{GIB, Known, MIB, Span};
use crate::request::{
    BuildRequest, InferenceRequest, Placement, Precision, ResourceRequest, TrainingRequest, params,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resource {
    Vram,
    Ram,
    Disk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    PartialCopy,
    DiskReserve,
    Weights,
    FrozenWeights,
    TrainableWeights,
    Gradients,
    OptimizerStates,
    KvCache,
    Activations,
    Logits,
    Adapters,
    RuntimeOverhead,
    Checkpoints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assumption {
    /// The result is written to `<out>.partial` and renamed on the same volume: one copy at a time.
    RenameOnSameVolume,
    DiskReserve,
    Int8NeedsRuntimeSupport,
    /// KV size follows config.json (window, shared layers); a runtime may allocate differently.
    KvFromConfig,
    ActivationsRough,
    RuntimeOverheadTypical,
    CpuOffloadIsSlow,
    NoAdapters,
    FullFineTuneF32AdamW,
    TrainingOnGpuUnverified,
    SingleGpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub part: Part,
    pub resource: Resource,
    pub size: Known<Span>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEstimate {
    pub items: Vec<Item>,
    pub assumptions: Vec<Assumption>,
}

/// The known part of a need, plus which parts could not be sized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Need {
    pub span: Span,
    pub unknown: Vec<Part>,
}

impl ResourceEstimate {
    pub fn need(&self, resource: Resource) -> Need {
        let mut need = Need { span: Span::default(), unknown: Vec::new() };
        for item in self.items.iter().filter(|i| i.resource == resource) {
            match &item.size {
                Known::Known(s) => need.span = need.span.plus(*s),
                Known::Unknown(_) => need.unknown.push(item.part),
            }
        }
        need
    }

    pub fn resources(&self) -> Vec<Resource> {
        let mut r: Vec<Resource> = self.items.iter().map(|i| i.resource).collect();
        r.sort();
        r.dedup();
        r
    }
}

fn scaled(count: u64, (low, high): (f64, f64)) -> Span {
    Span::between((count as f64 * low) as u64, (count as f64 * high) as u64)
}

fn device(p: Placement) -> Resource {
    if p == Placement::Gpu { Resource::Vram } else { Resource::Ram }
}

fn overhead(p: Placement) -> Span {
    if p == Placement::Gpu {
        Span::between(300 * MIB, 1536 * MIB)
    } else {
        Span::between(100 * MIB, 512 * MIB)
    }
}

pub fn estimate(req: &ResourceRequest) -> ResourceEstimate {
    match req {
        ResourceRequest::Build(b) => build(b),
        ResourceRequest::Inference(i) => inference(i),
        ResourceRequest::Training(t) => training(t),
    }
}

fn build(b: &BuildRequest) -> ResourceEstimate {
    let reserve = (256 * MIB).max(b.output_bytes / 50);
    ResourceEstimate {
        items: vec![
            Item {
                part: Part::PartialCopy,
                resource: Resource::Disk,
                size: Known::Known(Span::exact(b.output_bytes)),
            },
            Item {
                part: Part::DiskReserve,
                resource: Resource::Disk,
                size: Known::Known(Span::exact(reserve)),
            },
        ],
        assumptions: vec![Assumption::RenameOnSameVolume, Assumption::DiskReserve],
    }
}

fn inference(i: &InferenceRequest) -> ResourceEstimate {
    let dev = device(i.placement);
    let elem = if i.precision == Precision::F32 { 4 } else { 2 };
    let (ctx, batch) = (i.context, i.batch);
    let kv = i.attention.as_ref().map(|a| Span::exact(a.kv_cache_bytes(ctx, batch, elem)));
    let act = i.attention.as_ref().map(|a: &Attention| {
        let low = batch * ctx * a.hidden * elem * 2;
        let high = batch * ctx * (4 * a.hidden + 2 * a.intermediate) * elem + batch * ctx * a.vocab * 4;
        Span::between(low, high)
    });
    let mut assumptions =
        vec![Assumption::KvFromConfig, Assumption::ActivationsRough, Assumption::NoAdapters];
    assumptions.push(Assumption::RuntimeOverheadTypical);
    if i.precision == Precision::Int8 {
        assumptions.push(Assumption::Int8NeedsRuntimeSupport);
    }
    assumptions.push(if i.placement == Placement::Gpu {
        Assumption::SingleGpu
    } else {
        Assumption::CpuOffloadIsSlow
    });
    ResourceEstimate {
        items: vec![
            Item {
                part: Part::Weights,
                resource: dev,
                size: Known::Known(scaled(params(&i.components), i.precision.bytes_per_param())),
            },
            Item { part: Part::KvCache, resource: dev, size: kv },
            Item { part: Part::Activations, resource: dev, size: act },
            Item { part: Part::Adapters, resource: dev, size: Known::Known(Span::exact(0)) },
            Item { part: Part::RuntimeOverhead, resource: dev, size: Known::Known(overhead(i.placement)) },
        ],
        assumptions,
    }
}

fn training(t: &TrainingRequest) -> ResourceEstimate {
    let dev = device(t.placement);
    let (trainable, frozen) = (params(&t.trainable), params(&t.frozen));
    let (ctx, batch) = (t.context, t.batch);
    // Saved for backward, per token and layer, in f32: a range of hidden/intermediate-sized
    // tensors plus attention probabilities over the context.
    let act = t.attention.as_ref().map(|a| {
        let per =
            |kh: u64, ki: u64, ka: u64| (kh * a.hidden + ki * a.intermediate + ka * a.heads * ctx) * a.layers;
        Span::between(per(8, 3, 2) * batch * ctx * 4, per(16, 6, 3) * batch * ctx * 4)
    });
    let logits = t.attention.as_ref().map(|a| {
        let one = batch * ctx * a.vocab * 4;
        Span::between(2 * one, 3 * one)
    });
    let checkpoint = (trainable + frozen) * 4 + t.companion_bytes;
    let mut assumptions = vec![Assumption::FullFineTuneF32AdamW, Assumption::ActivationsRough];
    assumptions.push(Assumption::RuntimeOverheadTypical);
    assumptions.push(Assumption::RenameOnSameVolume);
    assumptions.push(if t.placement == Placement::Gpu {
        Assumption::TrainingOnGpuUnverified
    } else {
        Assumption::CpuOffloadIsSlow
    });
    let exact = |n: u64| Known::Known(Span::exact(n));
    ResourceEstimate {
        items: vec![
            Item { part: Part::FrozenWeights, resource: dev, size: exact(frozen * 4) },
            Item { part: Part::TrainableWeights, resource: dev, size: exact(trainable * 4) },
            Item { part: Part::Gradients, resource: dev, size: exact(trainable * 4) },
            Item { part: Part::OptimizerStates, resource: dev, size: exact(trainable * 8) },
            Item { part: Part::Activations, resource: dev, size: act },
            Item { part: Part::Logits, resource: dev, size: logits },
            Item { part: Part::RuntimeOverhead, resource: dev, size: Known::Known(overhead(t.placement)) },
            // Kept checkpoints plus the one being written to `.partial`.
            Item {
                part: Part::Checkpoints,
                resource: Resource::Disk,
                size: exact(checkpoint * (t.checkpoints_kept + 1)),
            },
            Item { part: Part::DiskReserve, resource: Resource::Disk, size: exact(GIB) },
        ],
        assumptions,
    }
}

#[cfg(test)]
#[path = "tests/estimate.rs"]
mod tests;
