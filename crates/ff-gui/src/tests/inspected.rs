//! Tests for module `inspected`.

use super::*;
use ff_core::TensorInfo;
use ff_io::Safetensors;

/// A checkpoint directory with two stacked blocks and one loose tensor.
fn checkpoint(name: &str, config: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ff-gui-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut m = Model::default();
    for name in ["model.layers.0.mlp.Up.weight", "model.layers.1.mlp.up.weight", "lm_head.weight"] {
        m.insert(TensorInfo { name: name.into(), dtype: DType::BF16, shape: vec![4, 2] });
    }
    Safetensors.write(&dir.join("model.safetensors"), &m, &mut |t| Ok(vec![0; t.byte_len()])).unwrap();
    if let Some(c) = config {
        fs::write(dir.join("config.json"), c).unwrap();
    }
    dir
}

#[test]
fn opens_directory_with_layout_and_config() {
    let dir = checkpoint("ok", Some(r#"{"num_hidden_layers":2}"#));
    let i = Inspected::open(&dir, &Safetensors).unwrap();
    assert_eq!(i.model.tensors.len(), 3);
    assert_eq!(i.layout.blocks.len(), 2);
    assert_eq!(i.layout.other, ["lm_head.weight"]);
    assert_eq!(i.params, 24);
    assert_eq!(i.dtypes[&DType::BF16], 3);
    let Config::Json { path, text } = &i.config else { panic!("{:?}", i.config) };
    assert_eq!(path, &dir.join("config.json"));
    assert!(text.contains("\"num_hidden_layers\": 2"), "{text}");
    assert_eq!(i.name, dir.file_name().unwrap().to_string_lossy());
    // A single file is named after its folder and shows the config next to it.
    let file = Inspected::open(&dir.join("model.safetensors"), &Safetensors).unwrap();
    assert_eq!(file.name, i.name);
    assert_eq!(file.config, i.config);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn search_ignores_case_and_blank_query_is_empty() {
    let dir = checkpoint("search", None);
    let i = Inspected::open(&dir, &Safetensors).unwrap();
    assert_eq!(i.search("UP"), ["model.layers.0.mlp.Up.weight", "model.layers.1.mlp.up.weight"]);
    assert!(i.search("  ").is_empty());
    assert!(i.search("nope").is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn config_missing_or_invalid_does_not_block_opening() {
    let dir = checkpoint("cfg", Some("{ not json"));
    let i = Inspected::open(&dir, &Safetensors).unwrap();
    let Config::Invalid { problem, detail } = &i.config else { panic!("{:?}", i.config) };
    assert_eq!(*problem, ConfigProblem::NotJson);
    // The parser's own message is kept for the technical details.
    assert!(detail.contains("config.json") && detail.contains("line 1"), "{detail}");
    fs::remove_file(dir.join("config.json")).unwrap();
    let file = Inspected::open(&dir.join("model.safetensors"), &Safetensors).unwrap();
    assert_eq!(file.config, Config::Missing);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn oversized_config_is_refused_without_reading_it_all() {
    let dir = checkpoint("big", None);
    // Sparse file: one byte over the limit costs no real disk space.
    fs::File::create(dir.join("config.json")).unwrap().set_len(MAX_CONFIG_LEN + 1).unwrap();
    let i = Inspected::open(&dir, &Safetensors).unwrap();
    let Config::Invalid { problem, detail } = &i.config else { panic!("{:?}", i.config) };
    assert_eq!(*problem, ConfigProblem::TooLarge);
    assert!(detail.contains("larger than"), "{detail}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn open_errors_are_reported_not_panicked() {
    let dir = checkpoint("bad", None);
    assert!(Inspected::open(&dir.join("missing.safetensors"), &Safetensors).is_err());
    let junk = dir.join("junk.safetensors");
    fs::write(&junk, b"definitely not safetensors").unwrap();
    let err = Inspected::open(&junk, &Safetensors).err().expect("junk must not open");
    assert!(err.to_string().contains("invalid model file"), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}
