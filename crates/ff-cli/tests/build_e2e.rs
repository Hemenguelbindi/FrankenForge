//! Full pipeline on two tiny safetensors models: recipe -> build -> verify -> lock.

use std::path::Path;

use ff_core::{DType, Model, ModelFormat, TensorInfo};
use ff_io::Safetensors;
use ff_surgery::{build, lock};

/// Three `[4, 4]` F32 layers, every byte set to `fill` so provenance is visible.
fn write_model(path: &Path, fill: u8) {
    let mut m = Model::default();
    for i in 0..3 {
        let name = format!("model.layers.{i}.mlp.weight");
        m.insert(TensorInfo { name, dtype: DType::F32, shape: vec![4, 4] });
    }
    Safetensors.write(path, &m, &mut |t| Ok(vec![fill; t.byte_len()])).unwrap();
}

const RECIPE: &str = r#"
name = "e2e"
base = "a"
[sources]
a = "a.safetensors"
b = "b.safetensors"
[[steps]]
op = "transplant"
from = "b"
src = "model.layers.2.mlp."
dst = "model.layers.0.mlp."
[[steps]]
op = "remove"
select = "model.layers.1."
[[steps]]
op = "rename"
src = "model.layers.2."
dst = "model.layers.1."
[[steps]]
op = "adapter"
name = "bridge.weight"
in_dim = 4
out_dim = 2
"#;

#[test]
fn transplant_remove_rename_adapter() {
    let dir = std::env::temp_dir().join(format!("ff-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_model(&dir.join("a.safetensors"), 1);
    write_model(&dir.join("b.safetensors"), 2);
    std::fs::write(dir.join("recipe.toml"), RECIPE).unwrap();

    let out = dir.join("out.safetensors");
    // The recipe has an adapter: without explicit opt-in the build would refuse.
    let report = build::build(&dir.join("recipe.toml"), &out, &Safetensors, true).unwrap();
    assert_eq!(report.tensors, 3);
    assert!(report.warnings.is_empty());
    assert!(report.experimental.is_some());

    let built = Safetensors.open(&out).unwrap();
    assert!(built.read("model.layers.0.mlp.weight").unwrap().iter().all(|&b| b == 2));
    assert!(built.read("model.layers.1.mlp.weight").unwrap().iter().all(|&b| b == 1));
    assert_eq!(built.model().tensor("bridge.weight").unwrap().shape, [2, 4]);
    assert_eq!(built.model().metadata["frankenforge.experimental"], "true");

    // The manifest (lock file) states explicitly that the result is not validated.
    let text = std::fs::read_to_string(&report.lock_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let e = &json["experimental"];
    assert_eq!(e["experimental"], true);
    assert_eq!(e["runtime_support"], "missing");
    assert_eq!(e["numerical_validation"], "not_performed");
    assert_eq!(e["quality_validation"], "not_performed");
    assert_eq!(e["tensors"], serde_json::json!(["bridge.weight"]));

    let lock: lock::Lock = serde_json::from_str(&text).unwrap();
    assert_eq!(lock.output_sha256, lock::hash_path(&out).unwrap());
    assert_eq!(lock.sources.len(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Without opt-in a build with an adapter refuses and leaves no output, `.partial` or lock.
#[test]
fn adapter_refused_by_default_without_partial_files() {
    let dir = std::env::temp_dir().join(format!("ff-e2e-adapter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_model(&dir.join("a.safetensors"), 1);
    write_model(&dir.join("b.safetensors"), 2);
    std::fs::write(dir.join("recipe.toml"), RECIPE).unwrap();

    for out in [dir.join("out.safetensors"), dir.join("out")] {
        let Err(err) = build::build(&dir.join("recipe.toml"), &out, &Safetensors, false) else {
            panic!("{}: build must refuse adapter without opt-in", out.display());
        };
        assert!(matches!(err, ff_core::Error::Experimental(_)), "{err}");
        assert!(err.to_string().contains("--allow-experimental-adapter"), "{err}");
        assert!(!out.exists());
        assert!(!lock::Lock::path_for(&out).exists());
    }
    assert!(!dir.join("out.partial").exists());
    // Only the input files are left in the directory.
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3);

    // A build without adapters does not mark the manifest.
    let plain = RECIPE.split("[[steps]]\nop = \"adapter\"").next().unwrap();
    std::fs::write(dir.join("plain.toml"), plain).unwrap();
    let out = dir.join("plain.safetensors");
    let report = build::build(&dir.join("plain.toml"), &out, &Safetensors, false).unwrap();
    assert!(report.experimental.is_none());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report.lock_path).unwrap()).unwrap();
    assert!(json.get("experimental").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Directory output: the gap after remove closes itself, config.json is updated,
/// the tokenizer is copied byte for byte, the source is unchanged.
#[test]
fn directory_output_renumbers_and_keeps_companions() {
    let dir = std::env::temp_dir().join(format!("ff-e2e-dir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let base = dir.join("base");
    std::fs::create_dir_all(&base).unwrap();
    write_model(&base.join("model.safetensors"), 1);
    write_model(&dir.join("b.safetensors"), 2);
    std::fs::write(base.join("config.json"), r#"{"num_hidden_layers": 3, "hidden_size": 4}"#).unwrap();
    std::fs::write(base.join("tokenizer.json"), r#"{"model": {"vocab": {"a": 0}}}"#).unwrap();
    let recipe = r#"
        name = "dir"
        base = "a"
        [sources]
        a = "base"
        b = "b.safetensors"
        [[steps]]
        op = "remove"
        select = "model.layers.1."
        [[steps]]
        op = "transplant"
        from = "b"
        src = "model.layers.2.mlp."
        dst = "model.layers.0.mlp."
    "#;
    std::fs::write(dir.join("recipe.toml"), recipe).unwrap();
    let source_hash = lock::hash_path(&base).unwrap();

    let out = dir.join("out");
    let report = build::build(&dir.join("recipe.toml"), &out, &Safetensors, false).unwrap();
    assert_eq!(report.tensors, 2);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    let built = Safetensors.open(&out.join("model.safetensors")).unwrap();
    assert!(built.read("model.layers.0.mlp.weight").unwrap().iter().all(|&b| b == 2));
    // Former layers.2 became layers.1.
    assert!(built.read("model.layers.1.mlp.weight").unwrap().iter().all(|&b| b == 1));
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["num_hidden_layers"], 2);
    assert_eq!(config["hidden_size"], 4);
    assert_eq!(
        std::fs::read(out.join("tokenizer.json")).unwrap(),
        std::fs::read(base.join("tokenizer.json")).unwrap()
    );
    assert_eq!(lock::hash_path(&base).unwrap(), source_hash);
    assert!(!dir.join("out.partial").exists());

    // Building into the same directory again refuses instead of overwriting it.
    assert!(build::build(&dir.join("recipe.toml"), &out, &Safetensors, false).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Regression: remove + transplant keep the block count but change block order and origin.
/// With an index-dependent config (`decoder_sparse_step`) this is refused before writing; a no-op passes.
#[test]
fn index_dependent_config_allows_only_noop() {
    let dir = std::env::temp_dir().join(format!("ff-e2e-index-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let base = dir.join("base");
    std::fs::create_dir_all(&base).unwrap();
    write_model(&base.join("model.safetensors"), 1);
    write_model(&dir.join("b.safetensors"), 2);
    let config = r#"{"num_hidden_layers": 3, "decoder_sparse_step": 2}"#;
    std::fs::write(base.join("config.json"), config).unwrap();
    let header = "name = \"i\"\nbase = \"a\"\n[sources]\na = \"base\"\nb = \"b.safetensors\"\n";
    // Remove block 1 and append a donor block: 3 blocks again.
    let steps = "[[steps]]\nop = \"remove\"\nselect = \"model.layers.1.\"\n\
                 [[steps]]\nop = \"transplant\"\nfrom = \"b\"\nsrc = \"model.layers.2.\"\ndst = \"model.layers.3.\"\n";
    std::fs::write(dir.join("shuffle.toml"), format!("{header}{steps}")).unwrap();
    std::fs::write(dir.join("noop.toml"), header).unwrap();
    let source_hash = lock::hash_path(&base).unwrap();

    // The block count really stays the same (3 -> 3); otherwise the test would exercise the old path.
    let recipe = ff_surgery::Recipe::load(&dir.join("shuffle.toml")).unwrap();
    let sources = ff_surgery::Sources::open(&recipe, &dir, &Safetensors).unwrap();
    let mut plan = ff_surgery::Plan::from_base("a", sources.get("a").unwrap().model());
    for step in &recipe.steps {
        ff_surgery::ops::apply(&mut plan, step, &sources).unwrap();
    }
    assert_eq!(plan.to_model().layout().blocks.len(), 3);

    for out in [dir.join("out"), dir.join("out.safetensors")] {
        let Err(err) = build::build(&dir.join("shuffle.toml"), &out, &Safetensors, false) else {
            panic!("{}: build must refuse", out.display());
        };
        assert!(matches!(err, ff_core::Error::UnsupportedConfig(_)), "{err}");
        assert!(err.to_string().contains("decoder_sparse_step"), "{err}");
        assert!(!out.exists());
        assert!(!lock::Lock::path_for(&out).exists());
    }
    assert!(!dir.join("out.partial").exists());
    assert_eq!(lock::hash_path(&base).unwrap(), source_hash);

    let out = dir.join("noop");
    build::build(&dir.join("noop.toml"), &out, &Safetensors, false).unwrap();
    assert_eq!(std::fs::read_to_string(out.join("config.json")).unwrap(), config);
    assert_eq!(lock::hash_path(&base).unwrap(), source_hash);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Config with per-layer `layer_types`: remove refuses and leaves nothing on disk,
/// while a build without steps copies that config as is.
#[test]
fn unsupported_config_fails_closed_but_noop_copies_it() {
    let dir = std::env::temp_dir().join(format!("ff-e2e-unsupported-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let base = dir.join("base");
    std::fs::create_dir_all(&base).unwrap();
    write_model(&base.join("model.safetensors"), 1);
    let config = r#"{"num_hidden_layers": 3, "layer_types": ["a", "b", "c"]}"#;
    std::fs::write(base.join("config.json"), config).unwrap();
    let header = "name = \"u\"\nbase = \"a\"\n[sources]\na = \"base\"\n";
    std::fs::write(dir.join("noop.toml"), header).unwrap();
    let remove = format!("{header}[[steps]]\nop = \"remove\"\nselect = \"model.layers.1.\"\n");
    std::fs::write(dir.join("remove.toml"), remove).unwrap();

    for out in [dir.join("out"), dir.join("out.safetensors")] {
        let Err(err) = build::build(&dir.join("remove.toml"), &out, &Safetensors, false) else {
            panic!("{}: build must refuse", out.display());
        };
        assert!(matches!(err, ff_core::Error::UnsupportedConfig(_)), "{err}");
        assert!(!out.exists());
        assert!(!lock::Lock::path_for(&out).exists());
    }
    assert!(!dir.join("out.partial").exists());

    let out = dir.join("noop");
    build::build(&dir.join("noop.toml"), &out, &Safetensors, false).unwrap();
    assert_eq!(std::fs::read_to_string(out.join("config.json")).unwrap(), config);
    std::fs::remove_dir_all(&dir).unwrap();
}
