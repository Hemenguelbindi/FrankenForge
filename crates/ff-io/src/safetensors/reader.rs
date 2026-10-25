use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ff_core::ports::out_of_range;
use ff_core::{Error, Model, Result, TensorSource};

use super::header::{self, MAX_HEADER_LEN};

/// One `.safetensors` file. Tensors are read lazily, one seek per call.
// shortcut: reopens the file on every read; switch to mmap if reads become the bottleneck.
pub struct SingleFile {
    path: PathBuf,
    model: Model,
    offsets: HashMap<String, (u64, u64)>,
    data_start: u64,
}

impl SingleFile {
    pub fn open(path: &Path) -> Result<Self> {
        let bad = |reason: String| Error::Format { path: path.to_owned(), reason };
        let mut f = fs::File::open(path).map_err(Error::io(path))?;
        let file_len = f.metadata().map_err(Error::io(path))?.len();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(Error::io(path))?;
        let header_len = u64::from_le_bytes(len);
        if header_len > MAX_HEADER_LEN || 8 + header_len > file_len {
            return Err(bad(format!("header length {header_len} out of range")));
        }
        let mut json = vec![0u8; header_len as usize];
        f.read_exact(&mut json).map_err(Error::io(path))?;
        let data_start = 8 + header_len;
        let parsed = header::parse(&json, file_len - data_start).map_err(bad)?;
        Ok(Self { path: path.to_owned(), model: parsed.model, offsets: parsed.offsets, data_start })
    }
}

impl TensorSource for SingleFile {
    fn model(&self) -> &Model {
        &self.model
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let &(start, end) = self.offsets.get(name).ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        self.read_range(name, 0, end - start)
    }

    fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
        let &(begin, end) = self.offsets.get(name).ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let size = end - begin;
        // `begin..end` was checked against the file when the header was parsed; the requested
        // range is checked against the tensor here, before allocating.
        let fits = start.checked_add(len).is_some_and(|stop| stop <= size);
        let buf_len =
            usize::try_from(len).ok().filter(|_| fits).ok_or_else(|| out_of_range(name, start, len, size))?;
        let mut f = fs::File::open(&self.path).map_err(Error::io(&self.path))?;
        f.seek(SeekFrom::Start(self.data_start + begin + start)).map_err(Error::io(&self.path))?;
        let mut buf = vec![0u8; buf_len];
        f.read_exact(&mut buf).map_err(Error::io(&self.path))?;
        Ok(buf)
    }
}
