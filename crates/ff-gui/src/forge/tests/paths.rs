//! Tests for module `forge::paths`: output names, resolving paths that do not exist yet, and
//! refusing outputs that would touch a source.

use std::fs;

use super::*;

/// A new empty directory, removed at the end of the test.
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ff-gui-paths-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn every_artifact_of_a_build_is_listed() {
    let out = Path::new("x").join("model");
    let names: Vec<String> =
        artifacts(&out).iter().map(|p| p.file_name().unwrap().to_string_lossy().into()).collect();
    assert_eq!(names, ["model.partial", "model.lock.json", "model.recipe.toml", "model.build.log", "model"]);
    assert_eq!(artifacts(&out)[1], ff_surgery::lock::Lock::path_for(&out), "same lock path as the core");
}

#[test]
fn only_one_plain_folder_name_is_valid() {
    assert!(valid_name("smollm2-forge"));
    for bad in ["", ".", "..", "a/b", "../a", " a", "a ", "/a"] {
        assert!(!valid_name(bad), "{bad:?}");
    }
    if cfg!(windows) {
        assert!(!valid_name("a\\b"));
        assert!(!valid_name("C:"));
    }
}

#[test]
fn resolves_a_missing_path_through_its_existing_ancestor() {
    let d = tmp("resolve");
    let real = d.canonicalize().unwrap();
    assert_eq!(resolve(&d.join("new").join("deeper")).unwrap(), real.join("new").join("deeper"));
    assert_eq!(resolve(&d.join("new").join("..").join("other")).unwrap(), real.join("other"));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_fresh_folder_next_to_the_sources_is_allowed() {
    let d = tmp("ok");
    let (base, donor) = (d.join("base"), d.join("donor"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&donor).unwrap();
    // The longer sibling name shares a string prefix with "base" but is not inside it.
    assert_eq!(check_output(&d.join("base-forge"), &[&base, &donor]), Ok(()));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn refuses_an_output_inside_a_source_even_if_it_does_not_exist() {
    let d = tmp("inside");
    let (base, donor) = (d.join("base"), d.join("donor"));
    fs::create_dir_all(&base).unwrap();
    fs::create_dir_all(&donor).unwrap();
    let sources = [base.as_path(), donor.as_path()];
    assert_eq!(check_output(&donor.join("out"), &sources), Err(OutputProblem::InsideSource));
    assert_eq!(check_output(&base.join("a").join("b"), &sources), Err(OutputProblem::InsideSource));
    // `..` cannot hide the target: d/donor/x/../out is inside donor.
    assert_eq!(
        check_output(&donor.join("x").join("..").join("out"), &sources),
        Err(OutputProblem::InsideSource)
    );
    if cfg!(windows) {
        let shouted = PathBuf::from(base.to_string_lossy().to_uppercase()).join("out");
        assert_eq!(check_output(&shouted, &sources), Err(OutputProblem::InsideSource));
    }
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn refuses_existing_paths_including_a_source_itself_and_leftovers() {
    let d = tmp("exists");
    let base = d.join("base");
    fs::create_dir_all(&base).unwrap();
    assert_eq!(check_output(&base, &[&base]), Err(OutputProblem::Exists(base.clone())));
    let partial = d.join("out.partial");
    fs::create_dir_all(&partial).unwrap();
    assert_eq!(check_output(&d.join("out"), &[&base]), Err(OutputProblem::Exists(partial)));
    fs::remove_dir_all(&d).unwrap();
}
