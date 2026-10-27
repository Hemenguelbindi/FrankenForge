//! Transplant of one middle block between two real SmolLM2-135M checkpoints (see
//! `examples/smollm2-transplant.toml`). Sources are only read; output goes to a new temporary directory.
//!
//! ```powershell
//! $env:FF_SMOLLM2_RECIPIENT = 'F:\models\SmolLM2-135M'
//! $env:FF_SMOLLM2_DONOR = 'F:\models\SmolLM2-135M-Instruct'
//! cargo test -p ff-cli --test real_smollm2_transplant -- --ignored --nocapture
//! ```

use std::fs;
use std::path::PathBuf;

use ff_core::ModelFormat;
use ff_io::Safetensors;
use ff_surgery::report::{Delta, FileCheck, Status};
use ff_surgery::{build, companion, inspect, lock};

const BLOCK: &str = "model.layers.15.";

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("set {var} to a model directory")))
}

fn companions(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| companion::is_companion(n))
        .collect();
    names.sort();
    names
}

#[test]
#[ignore = "needs FF_SMOLLM2_RECIPIENT and FF_SMOLLM2_DONOR"]
fn middle_block_from_donor_rest_from_recipient() {
    let (recipient, donor) = (env_path("FF_SMOLLM2_RECIPIENT"), env_path("FF_SMOLLM2_DONOR"));
    let (r_before, d_before) = (lock::hash_path(&recipient).unwrap(), lock::hash_path(&donor).unwrap());

    let dir = std::env::temp_dir().join(format!("ff-real-smollm2-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let recipe = dir.join("recipe.toml");
    // Literal TOML strings: backslashes in Windows paths need no escaping.
    let text = format!(
        "name = \"smollm2\"\nbase = \"recipient\"\n[sources]\nrecipient = '{}'\ndonor = '{}'\n\
         [[steps]]\nop = \"transplant\"\nfrom = \"donor\"\nsrc = \"{BLOCK}\"\ndst = \"{BLOCK}\"\n",
        recipient.display(),
        donor.display()
    );
    fs::write(&recipe, text).unwrap();
    let out = dir.join("out");
    let report = build::build(&recipe, &out, &Safetensors, false).unwrap();
    assert!(report.experimental.is_none());
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    assert!(!dir.join("out.partial").exists(), ".partial left behind");

    let r = Safetensors.open(&recipient).unwrap();
    let d = Safetensors.open(&donor).unwrap();
    let b = Safetensors.open(&out).unwrap();
    let blocks = b.model().layout().blocks;
    assert_eq!(blocks.len(), 30);
    assert_eq!(blocks.iter().map(|x| x.index).collect::<Vec<_>>(), (0..30).collect::<Vec<_>>());
    assert_eq!(r.model().tensors, b.model().tensors, "tensor names, dtypes or shapes differ");

    let (mut from_donor, mut from_recipient) = (0, 0);
    for name in b.model().tensors.keys() {
        let got = b.read(name).unwrap();
        if name.starts_with(BLOCK) {
            assert!(got == d.read(name).unwrap(), "{name}: differs from donor");
            assert!(
                got != r.read(name).unwrap(),
                "{name}: donor and recipient are identical, test proves nothing"
            );
            from_donor += 1;
        } else {
            assert!(got == r.read(name).unwrap(), "{name}: differs from recipient");
            from_recipient += 1;
        }
    }
    assert_eq!(from_donor, 9);

    let files = companions(&recipient);
    assert!(files.iter().any(|f| f == "config.json"), "{files:?}");
    assert!(files.iter().any(|f| f == "tokenizer.json"), "{files:?}");
    assert_eq!(companions(&out), files);
    for f in &files {
        assert!(
            fs::read(recipient.join(f)).unwrap() == fs::read(out.join(f)).unwrap(),
            "{f}: not the recipient's"
        );
    }

    let lock: lock::Lock = serde_json::from_str(&fs::read_to_string(&report.lock_path).unwrap()).unwrap();
    assert_eq!(report.lock_path, lock::Lock::path_for(&out));
    assert_eq!(lock.output_sha256, lock::hash_path(&out).unwrap());
    assert_eq!(lock.recipe_sha256, lock::hash_path(&recipe).unwrap());
    assert_eq!(lock.sources["recipient"].sha256, r_before);
    assert_eq!(lock.sources["donor"].sha256, d_before);
    assert!(lock.experimental.is_none());

    assert_eq!(lock::hash_path(&recipient).unwrap(), r_before, "recipient changed");
    assert_eq!(lock::hash_path(&donor).unwrap(), d_before, "donor changed");

    let text = inspect::report(b.model());
    assert!(text.contains("model.layers: 30 blocks"), "{text}");

    let rep = &report.report;
    assert_eq!(rep.stacks.len(), 1);
    assert_eq!((rep.stacks[0].blocks_before, rep.stacks[0].blocks_after), (30, 30));
    for block in &rep.stacks[0].blocks {
        if block.index == 15 {
            assert_eq!(block.status, Status::Replaced { alias: "donor".into(), from: 15 });
            assert_eq!((block.tensors, block.replaced.len()), (9, 9));
            let Some(Delta::Relative { value }) = block.delta else { panic!("{:?}", block.delta) };
            assert!(value > 0.0, "donor block 15 equals recipient block 15");
        } else {
            assert_eq!(block.status, Status::Unchanged, "block {}", block.index);
            assert!(block.delta.is_none());
        }
    }
    assert!(rep.removed_blocks.is_empty() && rep.removed_tensors.is_empty());
    assert_eq!(rep.file, FileCheck::Verified);
    assert_eq!(rep.model_behavior, "not_evaluated");
    let [json_path, md_path] = &report.report_paths;
    // Text, not a parsed Report: serde_json's default float parsing may differ in the last bit.
    assert_eq!(fs::read_to_string(json_path).unwrap(), rep.json());
    assert!(fs::read_to_string(md_path).unwrap().contains("model behavior: NOT EVALUATED"));
    println!("{}", rep.text());

    println!(
        "transplant: {from_donor} tensors of {BLOCK} from donor, {from_recipient} from recipient; \
         companions {files:?} from recipient; lock ok; sources unchanged"
    );
    fs::remove_dir_all(&dir).unwrap();
}
