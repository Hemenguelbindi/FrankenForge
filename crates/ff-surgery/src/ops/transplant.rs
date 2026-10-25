use ff_core::{Error, Result, TensorInfo, TensorSource};

use crate::{Payload, Plan, Planned};

/// Copies donor tensors under `src` into the plan under `dst`.
/// A replaced tensor must keep its dtype and shape; bridging sizes needs an adapter.
pub fn apply(plan: &mut Plan, donor: &dyn TensorSource, alias: &str, src: &str, dst: &str) -> Result<()> {
    let picked: Vec<&TensorInfo> = donor.model().select(src).collect();
    if picked.is_empty() {
        return Err(Error::Recipe(format!("transplant: `{alias}` has nothing under `{src}`")));
    }
    for t in picked {
        let name = format!("{dst}{}", &t.name[src.len()..]);
        if let Some(existing) = plan.tensors.get(&name)
            && !existing.info.same_layout(t)
        {
            return Err(Error::Incompatible(format!(
                "{name}: target is {:?} {:?}, donor {} is {:?} {:?}; add an adapter or pick a matching block",
                existing.info.dtype, existing.info.shape, t.name, t.dtype, t.shape
            )));
        }
        let payload = Payload::Source { alias: alias.into(), name: t.name.clone() };
        plan.insert(Planned { info: TensorInfo { name, ..t.clone() }, payload });
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/transplant.rs"]
mod tests;
