//! Tests for module `main` (command-line arguments).

use super::*;

fn args(list: &[&str]) -> (Option<PathBuf>, Option<Lang>) {
    parse_args(list.iter().map(OsString::from))
}

#[test]
fn lang_flag_and_path_in_any_order() {
    assert_eq!(args(&["--lang", "ru", "m.safetensors"]), (Some("m.safetensors".into()), Some(Lang::Ru)));
    assert_eq!(args(&["m", "--lang", "zh-CN"]), (Some("m".into()), Some(Lang::ZhCn)));
    // An unknown code is dropped instead of breaking startup.
    assert_eq!(args(&["--lang", "de"]), (None, None));
    assert_eq!(args(&[]), (None, None));
}
