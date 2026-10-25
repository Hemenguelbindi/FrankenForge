use std::collections::BTreeMap;
use std::path::Path;

use ff_core::{Error, ModelFormat, Result, TensorSource};

use crate::Recipe;

/// Opened source models keyed by recipe alias.
pub struct Sources {
    map: BTreeMap<String, Box<dyn TensorSource>>,
}

impl Sources {
    pub fn new(map: BTreeMap<String, Box<dyn TensorSource>>) -> Self {
        Self { map }
    }

    pub fn open(recipe: &Recipe, base_dir: &Path, format: &dyn ModelFormat) -> Result<Self> {
        let map = recipe
            .sources
            .iter()
            .map(|(alias, path)| Ok((alias.clone(), format.open(&base_dir.join(path))?)))
            .collect::<Result<_>>()?;
        Ok(Self { map })
    }

    pub fn get(&self, alias: &str) -> Result<&dyn TensorSource> {
        self.map
            .get(alias)
            .map(|s| s.as_ref())
            .ok_or_else(|| Error::Recipe(format!("unknown source alias `{alias}`")))
    }
}
