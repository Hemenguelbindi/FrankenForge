use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Recipe;
use crate::plan::Experimental;

/// Hashes of everything that went into a build, written next to the output
/// so the same model can be rebuilt and checked bit for bit.
#[derive(Debug, Serialize, Deserialize)]
pub struct Lock {
    pub tool_version: String,
    pub recipe_sha256: String,
    pub sources: BTreeMap<String, SourceLock>,
    pub output_sha256: String,
    /// Only for experimental builds (adapters): what was not validated in the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Experimental>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceLock {
    pub path: PathBuf,
    pub sha256: String,
}

impl Lock {
    pub fn create(
        recipe: &Recipe,
        recipe_path: &Path,
        output: &Path,
        experimental: Option<Experimental>,
    ) -> Result<Self> {
        let base_dir = Recipe::base_dir(recipe_path);
        let sources = recipe
            .sources
            .iter()
            .map(|(alias, path)| {
                let sha256 = hash_path(&base_dir.join(path))?;
                Ok((alias.clone(), SourceLock { path: path.clone(), sha256 }))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tool_version: env!("CARGO_PKG_VERSION").into(),
            recipe_sha256: hash_path(recipe_path)?,
            sources,
            output_sha256: hash_path(output)?,
            experimental,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).expect("plain struct");
        fs::write(path, json).map_err(Error::io(path))
    }

    /// `out.safetensors` -> `out.safetensors.lock.json`.
    pub fn path_for(output: &Path) -> PathBuf {
        let mut p = output.as_os_str().to_owned();
        p.push(".lock.json");
        p.into()
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of a file, or of every file in a directory (names and contents, sorted).
pub fn hash_path(path: &Path) -> Result<String> {
    let mut files = vec![path.to_owned()];
    if path.is_dir() {
        files = fs::read_dir(path)
            .map_err(Error::io(path))?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<_>>()
            .map_err(Error::io(path))?;
        files.retain(|p| p.is_file());
        files.sort();
    }
    let mut h = Sha256::new();
    for f in &files {
        if path.is_dir() {
            h.update(f.file_name().unwrap_or_default().as_encoded_bytes());
        }
        let mut file = fs::File::open(f).map_err(Error::io(f))?;
        std::io::copy(&mut file, &mut h).map_err(Error::io(f))?;
    }
    Ok(format!("{:x}", h.finalize()))
}
