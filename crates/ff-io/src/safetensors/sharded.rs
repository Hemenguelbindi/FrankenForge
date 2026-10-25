use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, Model, Result, TensorSource};

use super::reader::SingleFile;

/// A directory of `.safetensors` shards (`model-0000N-of-0000M`) seen as one model.
pub struct Sharded {
    model: Model,
    shards: Vec<SingleFile>,
    owner: HashMap<String, usize>,
}

impl Sharded {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(Error::io(dir))?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<_>>()
            .map_err(Error::io(dir))?;
        paths.retain(|p| p.extension().is_some_and(|x| x == "safetensors"));
        paths.sort();
        if paths.is_empty() {
            return Err(Error::Format { path: dir.to_owned(), reason: "no .safetensors files".into() });
        }

        let mut model = Model::default();
        let mut owner = HashMap::new();
        let mut shards = Vec::with_capacity(paths.len());
        for (i, path) in paths.iter().enumerate() {
            let shard = SingleFile::open(path)?;
            for info in shard.model().tensors.values() {
                if owner.insert(info.name.clone(), i).is_some() {
                    let reason = format!("tensor {} appears in more than one shard", info.name);
                    return Err(Error::Format { path: path.clone(), reason });
                }
                model.insert(info.clone());
            }
            model.metadata.extend(shard.model().metadata.clone());
            shards.push(shard);
        }
        Ok(Self { model, shards, owner })
    }
}

impl TensorSource for Sharded {
    fn model(&self) -> &Model {
        &self.model
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let &i = self.owner.get(name).ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        self.shards[i].read(name)
    }

    fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
        let &i = self.owner.get(name).ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        self.shards[i].read_range(name, start, len)
    }
}
