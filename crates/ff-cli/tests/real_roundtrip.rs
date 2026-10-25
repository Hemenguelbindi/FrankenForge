//! Round-trip on a real Safetensors model (a directory with config.json and tokenizer).
//! The model is not downloaded or modified; output goes to a new temporary directory.
//!
//! ```powershell
//! $env:FF_REAL_MODEL = 'F:\models\qwen3-moe-tiny-random'
//! cargo test -p ff-cli --test real_roundtrip -- --ignored --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use ff_core::{Error, ModelFormat};
use ff_io::Safetensors;
use ff_surgery::{build, companion, lock};
use serde_json::Value;

fn source() -> PathBuf {
    PathBuf::from(
        std::env::var_os("FF_REAL_MODEL").expect("set FF_REAL_MODEL to a safetensors model directory"),
    )
}

/// New empty directory for the recipe and the output.
fn workdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ff-real-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn write_recipe(dir: &Path, src: &Path, steps: &str) -> PathBuf {
    // Literal TOML string: backslashes in Windows paths need no escaping.
    let text = format!("name = \"real\"\nbase = \"m\"\n[sources]\nm = '{}'\n{steps}", src.display());
    let path = dir.join("recipe.toml");
    fs::write(&path, text).unwrap();
    path
}

fn companions(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| companion::is_companion(n))
        .collect();
    names.sort();
    names
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
#[ignore = "needs FF_REAL_MODEL"]
fn noop_roundtrip_preserves_tensors_config_tokenizer() {
    let src = source();
    let before = lock::hash_path(&src).unwrap();
    let dir = workdir("noop");
    let out = dir.join("out");
    let report = build::build(&write_recipe(&dir, &src, ""), &out, &Safetensors, false).unwrap();
    assert!(report.experimental.is_none());
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    let a = Safetensors.open(&src).unwrap();
    let b = Safetensors.open(&out.join("model.safetensors")).unwrap();
    assert_eq!(a.model().tensors, b.model().tensors, "tensor names, dtypes or shapes differ");
    assert_eq!(a.model().metadata, b.model().metadata);
    for name in a.model().tensors.keys() {
        assert!(a.read(name).unwrap() == b.read(name).unwrap(), "{name}: bytes differ");
    }

    let files = companions(&src);
    assert!(files.iter().any(|f| f == "config.json"), "{files:?}");
    assert!(files.iter().any(|f| f.starts_with("tokenizer")), "{files:?}");
    assert_eq!(companions(&out), files);
    for f in &files {
        assert!(fs::read(src.join(f)).unwrap() == fs::read(out.join(f)).unwrap(), "{f}: bytes differ");
    }

    let lock: lock::Lock = serde_json::from_str(&fs::read_to_string(&report.lock_path).unwrap()).unwrap();
    assert_eq!(lock.output_sha256, lock::hash_path(&out).unwrap());
    assert_eq!(lock.sources["m"].sha256, before);
    assert!(lock.experimental.is_none());
    assert_eq!(lock::hash_path(&src).unwrap(), before, "source changed");

    println!(
        "noop: {} tensors ({} params) byte-identical; {} files byte-identical: {files:?}; lock ok; source unchanged",
        report.tensors,
        report.params,
        files.len()
    );
    fs::remove_dir_all(&dir).unwrap();
}

/// Removing block 0: either renumbering + config.json, or a refusal that leaves nothing on disk.
#[test]
#[ignore = "needs FF_REAL_MODEL"]
fn remove_first_block_renumbers_or_fails_closed() {
    let src = source();
    let before = lock::hash_path(&src).unwrap();
    let base = Safetensors.open(&src).unwrap();
    let first = base.model().layout().blocks[0].clone();
    let n = base.model().layout().blocks.iter().filter(|b| b.stack == first.stack).count();
    assert!(n >= 2, "need at least two blocks in {}", first.stack);

    let dir = workdir("remove");
    let out = dir.join("out");
    let steps = format!("[[steps]]\nop = \"remove\"\nselect = \"{}\"\n", first.prefix);
    match build::build(&write_recipe(&dir, &src, &steps), &out, &Safetensors, false) {
        Ok(report) => {
            let built = Safetensors.open(&out.join("model.safetensors")).unwrap();
            let layout = built.model().layout();
            let blocks: Vec<_> = layout.blocks.iter().filter(|b| b.stack == first.stack).collect();
            let indices: Vec<usize> = blocks.iter().map(|b| b.index).collect();
            assert_eq!(indices, (0..n - 1).collect::<Vec<_>>(), "numbering not contiguous");
            for b in &blocks {
                for name in &b.tensors {
                    let old = format!("{}.{}.{}", first.stack, b.index + 1, &name[b.prefix.len()..]);
                    assert!(built.read(name).unwrap() == base.read(&old).unwrap(), "{name} != {old}");
                }
            }
            let (was, now) = (read_json(&src.join("config.json")), read_json(&out.join("config.json")));
            for (k, v) in was.as_object().unwrap() {
                let layer_key = ["num_hidden_layers", "n_layer", "num_layers"].contains(&k.as_str());
                if layer_key && v.as_u64() == Some(n as u64) {
                    assert_eq!(now[k], n as u64 - 1, "{k}");
                } else {
                    assert_eq!(&now[k], v, "{k}");
                }
            }
            println!(
                "remove: ok, {} -> {} blocks, indices {indices:?}; {} tensors",
                n,
                n - 1,
                report.tensors
            );
        }
        Err(Error::UnsupportedConfig(why)) => {
            println!("remove: fail-closed: {why}");
            assert!(!out.exists());
            assert!(!dir.join("out.partial").exists());
            assert!(!lock::Lock::path_for(&out).exists());
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
    assert_eq!(lock::hash_path(&src).unwrap(), before, "source changed");
    fs::remove_dir_all(&dir).unwrap();
}
