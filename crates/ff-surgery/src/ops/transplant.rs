use std::collections::BTreeSet;

use ff_core::{Error, Result, TensorInfo, TensorSource, layout};

use crate::{Payload, Plan, Planned};

/// Copies donor tensors under `src` into the plan under `dst`.
/// A replaced tensor must keep its dtype and shape; bridging sizes needs an adapter. A tensor
/// landing in a new block must match the same tensor of the stack's other blocks. Plan tensors
/// under `dst` the donor does not cover fail the step unless `keep_extra` allows a partial block.
pub fn apply(
    plan: &mut Plan,
    donor: &dyn TensorSource,
    alias: &str,
    src: &str,
    dst: &str,
    keep_extra: bool,
) -> Result<()> {
    let picked: Vec<(String, &TensorInfo)> =
        donor.model().select(src).map(|t| (format!("{dst}{}", &t.name[src.len()..]), t)).collect();
    if picked.is_empty() {
        return Err(Error::Recipe(format!("transplant: `{alias}` has nothing under `{src}`")));
    }
    for (name, t) in &picked {
        if let Some(existing) = plan.tensors.get(name) {
            if !existing.info.same_layout(t) {
                return Err(Error::Incompatible(format!(
                    "{name}: target is {:?} {:?}, donor {} is {:?} {:?}; add an adapter or pick a matching block",
                    existing.info.dtype, existing.info.shape, t.name, t.dtype, t.shape
                )));
            }
        } else if let Some(peer) = stack_peer(plan, name)
            && !peer.same_layout(t)
        {
            return Err(Error::Incompatible(format!(
                "{name}: new block, but {} in the same stack is {:?} {:?} and donor {} is {:?} {:?}; \
                 blocks of one stack must have the same tensor layout",
                peer.name, peer.dtype, peer.shape, t.name, t.dtype, t.shape
            )));
        }
    }
    let written: BTreeSet<&str> = picked.iter().map(|(n, _)| n.as_str()).collect();
    let extra: Vec<String> = plan.select(dst).into_iter().filter(|n| !written.contains(n.as_str())).collect();
    if !extra.is_empty() && !keep_extra {
        return Err(Error::Incompatible(format!(
            "transplant into `{dst}`: {extra:?} have no counterpart under `{alias}` `{src}` and would \
             silently stay; set keep_extra = true for a partial replacement or pick a matching block"
        )));
    }
    for (name, t) in picked {
        let payload = Payload::Source { alias: alias.into(), name: t.name.clone() };
        plan.insert(Planned { info: TensorInfo { name, ..t.clone() }, payload });
    }
    Ok(())
}

/// The same tensor (same name after the block prefix) in another block of `name`'s stack.
fn stack_peer<'a>(plan: &'a Plan, name: &str) -> Option<&'a TensorInfo> {
    let (stack, _, prefix) = layout::block_of(name)?;
    let suffix = &name[prefix.len()..];
    plan.tensors.values().map(|p| &p.info).find(|other| {
        layout::block_of(&other.name).is_some_and(|(s, _, p)| s == stack && &other.name[p.len()..] == suffix)
    })
}

#[cfg(test)]
#[path = "tests/transplant.rs"]
mod tests;
