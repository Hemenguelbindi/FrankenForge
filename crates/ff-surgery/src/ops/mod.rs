//! Recipe steps applied to a [`Plan`]. Each op checks compatibility up front
//! so a bad recipe fails before any weights are read.

mod adapter;
mod prefix;
mod transplant;

use ff_core::Result;

use crate::{Plan, Sources, Step};

pub use adapter::materialize;

pub fn apply(plan: &mut Plan, step: &Step, sources: &Sources) -> Result<()> {
    match step {
        Step::Transplant { from, src, dst, keep_extra } => {
            transplant::apply(plan, sources.get(from)?, from, src, dst, *keep_extra)
        }
        Step::Remove { select } => prefix::remove(plan, select),
        Step::Rename { src, dst } => prefix::rename(plan, src, dst),
        Step::Adapter { name, in_dim, out_dim, init, dtype } => {
            adapter::add(plan, name, *in_dim, *out_dim, *init, *dtype)
        }
    }
}
