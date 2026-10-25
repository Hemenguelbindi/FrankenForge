use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use ff_core::{Error, Model, Result, TensorInfo};

use super::header;

/// Writes to `<path>.partial` and renames on success, so a failed export
/// never leaves a truncated file at `path`.
pub fn write(
    path: &Path,
    model: &Model,
    fetch: &mut dyn FnMut(&TensorInfo) -> Result<Vec<u8>>,
) -> Result<()> {
    let tmp = path.with_extension("partial");
    match write_body(&tmp, model, fetch) {
        Ok(()) => fs::rename(&tmp, path).map_err(Error::io(path)),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn write_body(
    tmp: &Path,
    model: &Model,
    fetch: &mut dyn FnMut(&TensorInfo) -> Result<Vec<u8>>,
) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(tmp).map_err(Error::io(tmp))?);
    w.write_all(&header::serialize(model)).map_err(Error::io(tmp))?;
    for info in model.tensors.values() {
        let data = fetch(info)?;
        if data.len() != info.byte_len() {
            return Err(Error::Incompatible(format!(
                "{}: got {} bytes, expected {}",
                info.name,
                data.len(),
                info.byte_len()
            )));
        }
        w.write_all(&data).map_err(Error::io(tmp))?;
    }
    let file = w.into_inner().map_err(|e| Error::io(tmp)(e.into_error()))?;
    file.sync_all().map_err(Error::io(tmp))
}
