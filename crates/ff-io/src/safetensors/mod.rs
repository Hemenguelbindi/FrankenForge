mod header;
mod reader;
mod sharded;
mod writer;

use std::path::Path;

use ff_core::{Model, ModelFormat, Result, TensorInfo, TensorSource};

/// safetensors: a single file, or a directory of shards opened as one model.
pub struct Safetensors;

impl ModelFormat for Safetensors {
    fn open(&self, path: &Path) -> Result<Box<dyn TensorSource>> {
        if path.is_dir() {
            Ok(Box::new(sharded::Sharded::open(path)?))
        } else {
            Ok(Box::new(reader::SingleFile::open(path)?))
        }
    }

    fn file_name(&self) -> &'static str {
        "model.safetensors"
    }

    fn write(
        &self,
        path: &Path,
        model: &Model,
        fetch: &mut dyn FnMut(&TensorInfo) -> Result<Vec<u8>>,
    ) -> Result<()> {
        writer::write(path, model, fetch)
    }
}

#[cfg(test)]
#[path = "tests/safetensors.rs"]
mod tests;
