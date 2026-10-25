use ff_core::{Error, Result};

use crate::Plan;

pub fn remove(plan: &mut Plan, select: &str) -> Result<()> {
    let names = plan.select(select);
    if names.is_empty() {
        return Err(Error::Recipe(format!("remove: nothing matches `{select}`")));
    }
    for n in names {
        plan.tensors.remove(&n);
    }
    Ok(())
}

/// Moves every tensor under `src` to `dst`. All moved tensors are taken out
/// first, so shifting `layers.5.` onto a just-removed `layers.4.` works.
pub fn rename(plan: &mut Plan, src: &str, dst: &str) -> Result<()> {
    let names = plan.select(src);
    if names.is_empty() {
        return Err(Error::Recipe(format!("rename: nothing matches `{src}`")));
    }
    let moved: Vec<_> = names.iter().filter_map(|n| plan.tensors.remove(n)).collect();
    for mut p in moved {
        let name = format!("{dst}{}", &p.info.name[src.len()..]);
        if plan.tensors.contains_key(&name) {
            return Err(Error::Incompatible(format!("rename: `{name}` already exists")));
        }
        p.info.name = name;
        plan.insert(p);
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/prefix.rs"]
mod tests;
