//! Need versus the machine: one check per resource, an overall status and what could be reduced.
//! Unknown availability or an unsized part never yields `Fits`.

use crate::estimate::{Need, Resource, ResourceEstimate, estimate};
use crate::known::Known;
use crate::request::{Placement, ResourceRequest};
use crate::snapshot::HardwareSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightStatus {
    Fits,
    /// Fits only without the safety margin, or the range straddles the available amount.
    Tight,
    Insufficient,
    Unknown,
}

impl PreflightStatus {
    fn severity(self) -> u8 {
        match self {
            PreflightStatus::Fits => 0,
            PreflightStatus::Tight => 1,
            PreflightStatus::Unknown => 2,
            PreflightStatus::Insufficient => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub resource: Resource,
    pub need: Need,
    pub available: Known<u64>,
    pub status: PreflightStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advice {
    ReduceBatch,
    ReduceContext,
    FewerComponents,
    /// Moves weights and compute to RAM/CPU: an option with a speed cost, not a free fix.
    TryCpuOffload,
    FreeDisk,
    Reprobe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightReport {
    pub taken_at_unix: u64,
    pub estimate: ResourceEstimate,
    pub checks: Vec<Check>,
    pub status: PreflightStatus,
    pub advice: Vec<Advice>,
}

/// Share of the available amount kept free; the disk reserve is an explicit estimate part instead.
pub fn margin(resource: Resource) -> f64 {
    match resource {
        Resource::Vram => 0.10,
        Resource::Ram => 0.15,
        Resource::Disk => 0.0,
    }
}

pub fn check(need: &Need, available: &Known<u64>, margin: f64) -> PreflightStatus {
    let Known::Known(available) = available else { return PreflightStatus::Unknown };
    if need.span.low > *available {
        PreflightStatus::Insufficient
    } else if !need.unknown.is_empty() {
        PreflightStatus::Unknown
    } else if need.span.high as f64 <= *available as f64 * (1.0 - margin) {
        PreflightStatus::Fits
    } else {
        PreflightStatus::Tight
    }
}

fn available(req: &ResourceRequest, snap: &HardwareSnapshot, resource: Resource) -> Known<u64> {
    match resource {
        Resource::Vram => snap.gpu().map(|g| g.vram_free),
        Resource::Ram => snap.ram_available.clone(),
        Resource::Disk => {
            let dir = match req {
                ResourceRequest::Build(b) => Some(&b.out_dir),
                ResourceRequest::Training(t) => t.out_dir.as_ref(),
                ResourceRequest::Inference(_) => None,
            };
            match dir {
                Some(dir) => snap.disk_for(dir).map(|d| d.available),
                None => Known::unknown("no output folder chosen"),
            }
        }
    }
}

pub fn preflight(req: &ResourceRequest, snap: &HardwareSnapshot) -> PreflightReport {
    let estimate = estimate(req);
    let checks: Vec<Check> = estimate
        .resources()
        .into_iter()
        .map(|resource| {
            let need = estimate.need(resource);
            let available = available(req, snap, resource);
            let status = check(&need, &available, margin(resource));
            Check { resource, need, available, status }
        })
        .collect();
    let status =
        checks.iter().map(|c| c.status).max_by_key(|s| s.severity()).unwrap_or(PreflightStatus::Unknown);
    let advice = advice(req, &checks);
    PreflightReport { taken_at_unix: snap.taken_at_unix, estimate, checks, status, advice }
}

fn advice(req: &ResourceRequest, checks: &[Check]) -> Vec<Advice> {
    let mut out = Vec::new();
    let mut add = |a: Advice| {
        if !out.contains(&a) {
            out.push(a);
        }
    };
    for c in checks {
        if c.available.get().is_none() {
            add(Advice::Reprobe);
        }
        if !matches!(c.status, PreflightStatus::Insufficient | PreflightStatus::Tight) {
            continue;
        }
        if c.resource == Resource::Disk {
            add(Advice::FreeDisk);
            continue;
        }
        let (batch, many_components, placement) = match req {
            ResourceRequest::Build(_) => continue,
            ResourceRequest::Inference(i) => (i.batch, i.components.len() > 1, i.placement),
            // Full fine-tuning trains every component; dropping one is a different experiment.
            ResourceRequest::Training(t) => (t.batch, false, t.placement),
        };
        if batch > 1 {
            add(Advice::ReduceBatch);
        }
        add(Advice::ReduceContext);
        if many_components {
            add(Advice::FewerComponents);
        }
        if placement == Placement::Gpu && c.resource == Resource::Vram {
            add(Advice::TryCpuOffload);
        }
    }
    out
}

#[cfg(test)]
#[path = "tests/report.rs"]
mod tests;
