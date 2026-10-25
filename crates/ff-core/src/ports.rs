use std::path::Path;

use crate::{Error, Model, Result, TensorInfo};

/// An opened model whose tensors are read on demand.
pub trait TensorSource: Send + Sync {
    fn model(&self) -> &Model;

    /// Raw little-endian bytes of one tensor.
    fn read(&self, name: &str) -> Result<Vec<u8>>;

    /// Bytes `[start, start + len)` of one tensor, so a multi-GB tensor can be streamed in
    /// bounded memory. Required, not defaulted: a fallback reading the whole tensor per call
    /// would silently turn every streamed chunk into a full read.
    fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>>;
}

/// `read_range` for sources that hold a tensor's bytes in memory anyway (tests, small formats).
pub fn slice_range(all: &[u8], name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
    let range = usize::try_from(start)
        .ok()
        .zip(usize::try_from(len).ok())
        .and_then(|(s, l)| Some(s..s.checked_add(l)?))
        .filter(|r| r.end <= all.len())
        .ok_or_else(|| out_of_range(name, start, len, all.len() as u64))?;
    Ok(all[range].to_vec())
}

/// Error for a byte range that does not fit inside a `size`-byte tensor.
pub fn out_of_range(name: &str, start: u64, len: u64, size: u64) -> Error {
    Error::Incompatible(format!("{name}: bytes {start}+{len} are outside the {size}-byte tensor"))
}

/// A checkpoint format (safetensors, later GGUF, ...). Adapters live in `ff-io`.
pub trait ModelFormat {
    fn open(&self, path: &Path) -> Result<Box<dyn TensorSource>>;

    /// Name of the weights file inside an output directory (`model.safetensors`); its extension
    /// tells `-o file.<ext>` apart from `-o dir`.
    fn file_name(&self) -> &'static str;

    /// Writes `model` to `path`, pulling each tensor's bytes from `fetch`
    /// in `model.tensors` order so only one tensor is in memory at a time.
    fn write(
        &self,
        path: &Path,
        model: &Model,
        fetch: &mut dyn FnMut(&TensorInfo) -> Result<Vec<u8>>,
    ) -> Result<()>;
}

#[cfg(test)]
#[path = "tests/ports.rs"]
mod tests;
