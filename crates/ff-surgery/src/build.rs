use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use ff_core::{Error, Model, ModelFormat, Result, TensorInfo};

use crate::companion::Companion;
use crate::lock::{Lock, sha256};
use crate::plan::Experimental;
use crate::{Payload, Plan, Recipe, Sources, ops};

pub struct BuildReport {
    pub tensors: usize,
    pub params: usize,
    pub warnings: Vec<String>,
    pub lock_path: PathBuf,
    /// `Some`: the build is experimental and the result is not a validated working model.
    pub experimental: Option<Experimental>,
}

/// Opens the sources and applies every step. Touches metadata only.
pub fn plan(recipe: &Recipe, base_dir: &Path, format: &dyn ModelFormat) -> Result<(Sources, Plan)> {
    let sources = Sources::open(recipe, base_dir, format)?;
    let base = sources.get(&recipe.base)?.model();
    let mut plan = Plan::from_base(&recipe.base, base);
    for (i, step) in recipe.steps.iter().enumerate() {
        ops::apply(&mut plan, step, &sources).map_err(|e| Error::Recipe(format!("step {}: {e}", i + 1)))?;
    }
    plan.compact();
    plan.metadata.extend(recipe.metadata.clone());
    // `sources.get` above already checked that base is in recipe.sources.
    let base_path = base_dir.join(&recipe.sources[&recipe.base]);
    let blocks_changed = plan.blocks_changed(&recipe.base, base);
    plan.companion = Companion::load(&base_path, &base.layout(), &plan.to_model().layout(), blocks_changed)?;
    Ok((sources, plan))
}

/// Non-fatal findings, e.g. layer-index gaps left by a removal.
pub fn warnings(plan: &Plan) -> Vec<String> {
    plan.to_model().layout().gaps()
}

/// Plans, writes, re-reads to verify, then writes `<out>.lock.json`.
/// An `out` with the format's extension is a single weights file; anything else is a directory with weights, config.json and tokenizer.
/// A build with adapters and no `allow_experimental` refuses before writing anything.
pub fn build(
    recipe_path: &Path,
    out: &Path,
    format: &dyn ModelFormat,
    allow_experimental: bool,
) -> Result<BuildReport> {
    let recipe = Recipe::load(recipe_path)?;
    let (sources, mut plan) = plan(&recipe, Recipe::base_dir(recipe_path), format)?;
    let experimental = plan.experimental();
    if let Some(e) = &experimental {
        if !allow_experimental {
            return Err(Error::Experimental(format!(
                "adapter tensors {:?} have no runtime support and no numerical or quality validation; \
                 refusing to build without explicit opt-in (--allow-experimental-adapter)",
                e.tensors
            )));
        }
        // Marker inside the weights file itself, so the result is never passed off as a working model.
        plan.metadata.insert("frankenforge.experimental".into(), "true".into());
    }
    let model = plan.to_model();
    let mut warnings = warnings(&plan);

    if out.extension() == Path::new(format.file_name()).extension() {
        if plan.companion.dir.is_some() {
            warnings.push(
                "single-file output: config.json and tokenizer are not written; pass a directory to -o"
                    .into(),
            );
        }
        write_weights(format, out, &plan, &sources, &model)?;
    } else {
        write_dir(format, out, &plan, &sources, &model)?;
    }

    let lock_path = Lock::path_for(out);
    Lock::create(&recipe, recipe_path, out, experimental.clone())?.save(&lock_path)?;
    Ok(BuildReport {
        tensors: model.tensors.len(),
        params: model.param_count(),
        warnings,
        lock_path,
        experimental,
    })
}

/// Builds the directory in `<out>.partial` and renames it after verification. An existing `out`
/// is never touched: overwriting a directory could delete someone else's files.
fn write_dir(
    format: &dyn ModelFormat,
    out: &Path,
    plan: &Plan,
    sources: &Sources,
    model: &Model,
) -> Result<()> {
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(".partial");
    let tmp = PathBuf::from(tmp);
    for p in [out, &tmp] {
        if p.exists() {
            return Err(Error::io(p)(ErrorKind::AlreadyExists.into()));
        }
    }
    fs::create_dir_all(&tmp).map_err(Error::io(&tmp))?;
    let result = write_weights(format, &tmp.join(format.file_name()), plan, sources, model)
        .and_then(|()| plan.companion.write(&tmp))
        .and_then(|()| fs::rename(&tmp, out).map_err(Error::io(out)));
    if result.is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    result
}

fn write_weights(
    format: &dyn ModelFormat,
    path: &Path,
    plan: &Plan,
    sources: &Sources,
    model: &Model,
) -> Result<()> {
    let mut written = HashMap::new();
    format.write(path, model, &mut |info| {
        let data = fetch(plan, sources, info)?;
        written.insert(info.name.clone(), sha256(&data));
        Ok(data)
    })?;
    verify(format, path, plan, &written)
}

fn fetch(plan: &Plan, sources: &Sources, info: &TensorInfo) -> Result<Vec<u8>> {
    match &plan.tensors[&info.name].payload {
        Payload::Source { alias, name } => sources.get(alias)?.read(name),
        Payload::Generated(init) => Ok(ops::materialize(info, *init)),
    }
}

/// Reopens the output and checks every tensor's layout and content hash.
fn verify(
    format: &dyn ModelFormat,
    out: &Path,
    plan: &Plan,
    written: &HashMap<String, String>,
) -> Result<()> {
    let back = format.open(out)?;
    if back.model().tensors.len() != plan.tensors.len() {
        return Err(Error::Verify(format!(
            "expected {} tensors, found {}",
            plan.tensors.len(),
            back.model().tensors.len()
        )));
    }
    for (name, p) in &plan.tensors {
        if !back.model().tensor(name)?.same_layout(&p.info) {
            return Err(Error::Verify(format!("{name}: layout changed on disk")));
        }
        if sha256(&back.read(name)?) != written[name] {
            return Err(Error::Verify(format!("{name}: content hash mismatch")));
        }
    }
    Ok(())
}
