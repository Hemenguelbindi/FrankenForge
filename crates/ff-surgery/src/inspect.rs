use std::collections::BTreeMap;
use std::fmt::Write;

use ff_core::{Block, DType, Model};

/// Human-readable summary: totals, each layer stack with block 0 as a
/// template (shapes are usually identical across a stack), then the rest.
pub fn report(model: &Model) -> String {
    let mut out = String::new();
    let mut dtypes: BTreeMap<DType, usize> = BTreeMap::new();
    for t in model.tensors.values() {
        *dtypes.entry(t.dtype).or_default() += 1;
    }
    let _ = writeln!(
        out,
        "{} tensors, {} parameters, dtypes {dtypes:?}",
        model.tensors.len(),
        model.param_count()
    );

    let layout = model.layout();
    let mut stacks: BTreeMap<&str, Vec<&Block>> = BTreeMap::new();
    for b in &layout.blocks {
        stacks.entry(&b.stack).or_default().push(b);
    }
    for (stack, blocks) in stacks {
        let (first, last) = (blocks[0], blocks[blocks.len() - 1]);
        let _ = writeln!(out, "\n{stack}: {} blocks, indices {}..={}", blocks.len(), first.index, last.index);
        for name in &first.tensors {
            let t = &model.tensors[name];
            let _ = writeln!(
                out,
                "  [{}].{:<40} {:?} {:?}",
                first.index,
                &name[first.prefix.len()..],
                t.dtype,
                t.shape
            );
        }
    }

    if !layout.other.is_empty() {
        let _ = writeln!(out, "\nother:");
        for name in &layout.other {
            let t = &model.tensors[name];
            let _ = writeln!(out, "  {name:<44} {:?} {:?}", t.dtype, t.shape);
        }
    }
    out
}
