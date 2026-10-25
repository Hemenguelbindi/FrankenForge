use ff_core::{DType, Error, Result, TensorInfo};

use crate::{AdapterInit, Payload, Plan, Planned};

pub fn add(
    plan: &mut Plan,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    init: AdapterInit,
    dtype: DType,
) -> Result<()> {
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Recipe(format!("adapter {name}: dtype {dtype:?} is not F32, F16 or BF16")));
    }
    if in_dim == 0 || out_dim == 0 {
        return Err(Error::Recipe(format!("adapter {name}: dimensions must be non-zero")));
    }
    if plan.tensors.contains_key(name) {
        return Err(Error::Incompatible(format!("adapter {name}: tensor already exists")));
    }
    let info = TensorInfo { name: name.into(), dtype, shape: vec![out_dim, in_dim] };
    plan.insert(Planned { info, payload: Payload::Generated(init) });
    Ok(())
}

/// Bytes for a generated `[rows, cols]` adapter in the dtypes `add` accepts.
pub fn materialize(info: &TensorInfo, init: AdapterInit) -> Vec<u8> {
    let mut out = vec![0u8; info.byte_len()];
    if init == AdapterInit::Zeros {
        return out;
    }
    let one: Vec<u8> = match info.dtype {
        DType::F32 => 1f32.to_le_bytes().to_vec(),
        DType::F16 => vec![0x00, 0x3C],
        DType::BF16 => vec![0x80, 0x3F],
        other => unreachable!("adapter dtype {other:?} rejected in add"),
    };
    let (rows, cols) = (info.shape[0], info.shape[1]);
    for i in 0..rows.min(cols) {
        let at = (i * cols + i) * one.len();
        out[at..at + one.len()].copy_from_slice(&one);
    }
    out
}

#[cfg(test)]
#[path = "tests/adapter.rs"]
mod tests;
