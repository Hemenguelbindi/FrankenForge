//! Tests for module `forge::build` on tiny models written in Rust: success with every file next to
//! the output, refusal before any write, and a core failure that keeps its diagnosis.

use std::fs;

use ff_core::{DType, Model, ModelFormat, TensorInfo};
use ff_surgery::Recipe;
use ff_surgery::lock::hash_path;

use super::*;
use crate::forge::analysis::BlockId;
use crate::forge::state::Pair;

/// A model folder: three `[4]` F32 layers with every byte = `fill`, plus config and tokenizer.
fn model_dir(dir: &Path, fill: u8) {
    fs::create_dir_all(dir).unwrap();
    let mut m = Model::default();
    for i in 0..3 {
        m.insert(TensorInfo {
            name: format!("model.layers.{i}.mlp.weight"),
            dtype: DType::F32,
            shape: vec![4],
        });
    }
    Safetensors.write(&dir.join("model.safetensors"), &m, &mut |t| Ok(vec![fill; t.byte_len()])).unwrap();
    fs::write(dir.join("config.json"), r#"{"num_hidden_layers": 3}"#).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"model": {"vocab": {"a": 0}}}"#).unwrap();
}

/// New base (fill 1) and donor (fill 2) in a fresh directory.
fn setup(name: &str) -> (PathBuf, Pair) {
    let d = std::env::temp_dir().join(format!("ff-gui-build-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    let pair = Pair { base: d.join("base"), donor: d.join("donor") };
    model_dir(&pair.base, 1);
    model_dir(&pair.donor, 2);
    (d, pair)
}

fn request(pair: &Pair, index: usize, out: PathBuf) -> BuildRequest {
    let blocks = vec![BlockId { stack: "model.layers".into(), index }];
    let preview = Preview {
        from_base: 2,
        from_donor: 1,
        notes: Vec::new(),
        output_bytes: 1024,
        report: crate::forge::state::empty_report(),
    };
    BuildRequest { selection: Selection { pair: pair.clone(), blocks }, out, preview }
}

fn hashes(pair: &Pair) -> (String, String) {
    (hash_path(&pair.base).unwrap(), hash_path(&pair.donor).unwrap())
}

#[test]
fn a_known_disk_shortage_refuses_before_any_write() {
    let (d, pair) = setup("nospace");
    let out = d.join("forged");
    let mut req = request(&pair, 1, out.clone());
    // No real volume has an exabyte free: the fresh check right before writing refuses.
    req.preview.output_bytes = 1 << 60;
    let err = run(&req).unwrap().unwrap_err();
    assert!(
        matches!(err, BuildError::Refused(OutputProblem::NoSpace { need, .. }) if need >= 1 << 60),
        "{err:?}"
    );
    for p in paths::artifacts(&out) {
        assert!(!p.exists(), "{} was written", p.display());
    }
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_build_writes_a_verified_model_with_recipe_lock_and_log() {
    let (d, pair) = setup("ok");
    let before = hashes(&pair);
    let out = d.join("forged");
    let built = run(&request(&pair, 1, out.clone())).unwrap().unwrap();

    assert_eq!(built.out, out);
    assert_eq!(built.lock, Lock::path_for(&out));
    assert_eq!((built.tensors, built.planned.from_donor), (3, 1));
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    let model = Safetensors.open(&out).unwrap();
    for (i, fill) in [(0, 1), (1, 2), (2, 1)] {
        let bytes = model.read(&format!("model.layers.{i}.mlp.weight")).unwrap();
        assert!(bytes.iter().all(|&b| b == fill), "block {i} should come from fill {fill}");
    }
    assert_eq!(fs::read(out.join("config.json")).unwrap(), fs::read(pair.base.join("config.json")).unwrap());

    let recipe = Recipe::load(&built.recipe).unwrap();
    assert_eq!(recipe.sources["donor"], pair.donor);
    let lock: Lock = serde_json::from_str(&fs::read_to_string(&built.lock).unwrap()).unwrap();
    assert_eq!(lock.recipe_sha256, hash_path(&built.recipe).unwrap());
    assert_eq!(lock.output_sha256, hash_path(&out).unwrap());
    let log = fs::read_to_string(built.log.as_ref().unwrap()).unwrap();
    assert!(log.contains("result: OK") && log.contains("model.layers.1."), "{log}");
    assert!(!paths::sibling(&out, ".partial").exists());
    assert_eq!(hashes(&pair), before, "sources changed");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn an_output_that_appeared_after_the_readiness_check_is_refused_before_any_write() {
    let (d, pair) = setup("appeared");
    let out = d.join("forged");
    // Someone else creates the folder between the cached check and the click.
    fs::create_dir_all(&out).unwrap();
    let err = run(&request(&pair, 1, out.clone())).unwrap().unwrap_err();
    assert_eq!(err, BuildError::Refused(OutputProblem::Exists(out.clone())));
    assert!(!paths::sibling(&out, paths::RECIPE).exists() && !paths::sibling(&out, paths::LOG).exists());

    // A leftover recipe from an earlier failed run blocks the name too.
    let other = d.join("other");
    fs::write(paths::sibling(&other, paths::RECIPE), "x").unwrap();
    let err = run(&request(&pair, 1, other.clone())).unwrap().unwrap_err();
    assert_eq!(err, BuildError::Refused(OutputProblem::Exists(paths::sibling(&other, paths::RECIPE))));
    assert!(!other.exists());
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn an_output_inside_a_source_is_refused_and_the_source_stays_unchanged() {
    let (d, pair) = setup("inside");
    let before = hashes(&pair);
    for out in [pair.base.join("forged"), pair.donor.join("x").join("forged")] {
        let err = run(&request(&pair, 1, out.clone())).unwrap().unwrap_err();
        assert_eq!(err, BuildError::Refused(OutputProblem::InsideSource), "{}", out.display());
    }
    assert_eq!(hashes(&pair), before);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_core_failure_keeps_recipe_and_log_and_leaves_no_model() {
    let (d, pair) = setup("fails");
    let before = hashes(&pair);
    let out = d.join("forged");
    // The donor has no block 7: planning fails inside build::build.
    let err = run(&request(&pair, 7, out.clone())).unwrap().unwrap_err();
    let BuildError::Failed { failure, recipe, log } = err else { panic!("expected a core failure") };
    assert!(failure.detail.contains("model.layers.7."), "{}", failure.detail);
    assert_eq!(recipe, Some(paths::sibling(&out, paths::RECIPE)));
    let log = fs::read_to_string(log.unwrap()).unwrap();
    assert!(log.contains("result: FAILED") && log.contains(&failure.detail), "{log}");
    assert!(!out.exists() && !paths::sibling(&out, ".partial").exists() && !Lock::path_for(&out).exists());
    assert_eq!(hashes(&pair), before);
    fs::remove_dir_all(&d).unwrap();
}
