//! Tests for module `forge::jobs` on two real checkpoints (only headers and tokenizer.json are read).
//!
//! ```powershell
//! $env:FF_SMOLLM2_RECIPIENT = '<models>\SmolLM2-135M'
//! $env:FF_SMOLLM2_DONOR = '<models>\SmolLM2-135M-Instruct'
//! cargo test -p ff-gui forge::jobs -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use ff_surgery::compare::TokenizerMatch;

use super::*;
use crate::forge::analysis::BlockStatus;
use crate::forge::state::Forge;

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("set {var} to a model directory")))
}

#[test]
#[ignore = "needs FF_SMOLLM2_RECIPIENT and FF_SMOLLM2_DONOR"]
fn smollm2_block_15_previews_9_donor_and_263_base_tensors() {
    let pair = Pair { base: env_path("FF_SMOLLM2_RECIPIENT"), donor: env_path("FF_SMOLLM2_DONOR") };
    let analysis = analyze(&pair).unwrap();
    assert_eq!(analysis.report.tokenizer.status, TokenizerMatch::Identical);
    assert_eq!(analysis.rows.len(), 30);
    assert!(analysis.rows.iter().all(|r| r.status == BlockStatus::Compatible));

    let mut f = Forge::default();
    f.set_base(pair.base.clone());
    f.set_donor(pair.donor.clone());
    f.apply_analysis(pair, Ok(analysis));
    let block_15 = f.analysis().unwrap().as_ref().unwrap().rows[15].id.clone();
    assert_eq!(block_15.prefix(), "model.layers.15.");
    f.toggle(&block_15);
    let p = preview(&f.selection().unwrap()).unwrap();
    assert_eq!((p.from_donor, p.from_base), (9, 263));
    assert!(p.notes.is_empty(), "{:?}", p.notes);
    println!("preview: {} from donor, {} from base", p.from_donor, p.from_base);
}

/// The GUI build job end to end on the real models; output goes to a new temporary directory.
#[test]
#[ignore = "needs FF_SMOLLM2_RECIPIENT and FF_SMOLLM2_DONOR"]
fn smollm2_block_15_builds_verifies_and_reopens() {
    use ff_core::ModelFormat;
    use ff_surgery::{inspect, lock};

    use crate::forge::build::{BuildRequest, run};

    let pair = Pair { base: env_path("FF_SMOLLM2_RECIPIENT"), donor: env_path("FF_SMOLLM2_DONOR") };
    let before = (lock::hash_path(&pair.base).unwrap(), lock::hash_path(&pair.donor).unwrap());
    let mut f = Forge::default();
    f.set_base(pair.base.clone());
    f.set_donor(pair.donor.clone());
    f.apply_analysis(pair.clone(), analyze(&pair));
    let block_15 = f.analysis().unwrap().as_ref().unwrap().rows[15].id.clone();
    f.toggle(&block_15);
    let sel = f.selection().unwrap();
    f.apply_preview(sel.clone(), preview(&sel));

    let dir = std::env::temp_dir().join(format!("ff-gui-real-build-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    f.set_out_dir(dir.clone());
    f.set_out_name("smollm2-block15".into());
    let req: BuildRequest = f.begin_build().expect("ready to build");
    let built = run(&req).unwrap().expect("build succeeds");
    f.finish_build(req, Ok(Ok(built)));
    let crate::forge::state::BuildStatus::Done(Ok(built)) = f.build_status() else { panic!("not built") };

    let (r, d) = (Safetensors.open(&pair.base).unwrap(), Safetensors.open(&pair.donor).unwrap());
    let out = Safetensors.open(&built.out).unwrap();
    assert_eq!(out.model().tensors, r.model().tensors, "names, dtypes or shapes differ from the base");
    let (mut from_donor, mut from_base) = (0, 0);
    for name in out.model().tensors.keys() {
        let bytes = out.read(name).unwrap();
        if name.starts_with("model.layers.15.") {
            assert!(bytes == d.read(name).unwrap(), "{name}: not the donor's");
            from_donor += 1;
        } else {
            assert!(bytes == r.read(name).unwrap(), "{name}: not the base's");
            from_base += 1;
        }
    }
    assert_eq!((from_donor, from_base), (9, 263));
    assert_eq!((built.planned.from_donor, built.planned.from_base), (9, 263));

    let lock: lock::Lock = serde_json::from_str(&std::fs::read_to_string(&built.lock).unwrap()).unwrap();
    assert_eq!(lock.output_sha256, lock::hash_path(&built.out).unwrap());
    assert_eq!((lock.sources["base"].sha256.clone(), lock.sources["donor"].sha256.clone()), before);
    assert!(lock.experimental.is_none());
    assert!(std::fs::read_to_string(built.log.as_ref().unwrap()).unwrap().contains("result: OK"));
    assert!(!dir.join("smollm2-block15.partial").exists());
    assert!(inspect::report(out.model()).contains("model.layers: 30 blocks"));
    let after = (lock::hash_path(&pair.base).unwrap(), lock::hash_path(&pair.donor).unwrap());
    assert_eq!(after, before, "sources changed");
    println!(
        "built {}: {from_donor} tensors from donor, {from_base} from base; lock, log, verify ok; sources unchanged",
        built.out.display()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
