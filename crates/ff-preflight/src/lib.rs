//! Hardware Preflight: an estimate of the disk, RAM and VRAM an operation needs, compared with
//! a snapshot of the machine. Everything is an estimate with visible parts and assumptions, never
//! proof that a model loads or how fast it runs. Unknown facts stay Unknown; they are never
//! turned into zero or into "fits".
//!
//! `probe` is the only module that touches the system; the rest is pure and tested on synthetic
//! snapshots.

pub mod attention;
pub mod estimate;
pub mod known;
pub mod probe;
pub mod report;
pub mod request;
pub mod snapshot;

pub use estimate::{Assumption, Item, Need, Part, Resource, ResourceEstimate, estimate};
pub use known::{Known, Span};
pub use report::{Advice, Check, PreflightReport, PreflightStatus, preflight};
pub use request::{
    BuildRequest, Component, InferenceRequest, Placement, Precision, ResourceRequest, TrainingRequest,
};
pub use snapshot::{DiskInfo, GpuInfo, HardwareSnapshot};
