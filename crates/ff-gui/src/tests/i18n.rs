//! Tests for module `i18n`: language choice, fallback, missing keys and locale completeness.

use std::collections::BTreeSet;

use super::*;
use crate::components::Kind;
use crate::failure::FailureKind;
use crate::inspected::ConfigProblem;
use crate::inspector::Mode;
use crate::learn::{Explanation, STEPS};
use crate::presentation::Presentation;
use crate::semantics::{Basis, DIM_KEYS, Group, Role};
use crate::structure::{group_keys, role_keys};

/// Message ids defined in an FTL source (`id = ...` at the start of a line).
fn ids(source: &str) -> BTreeSet<&str> {
    source
        .lines()
        .filter_map(|l| l.split_once(" ="))
        .map(|(id, _)| id)
        .filter(|id| {
            !id.is_empty() && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        })
        .collect()
}

/// Keys passed to `.t("...")` / `.ta("...")` in a source file.
fn keys_used(source: &str) -> Vec<&str> {
    let mut keys = Vec::new();
    for call in [".t(\"", ".ta(\""] {
        let mut rest = source;
        while let Some(at) = rest.find(call) {
            rest = &rest[at + call.len()..];
            if let Some(end) = rest.find('"') {
                keys.push(&rest[..end]);
            }
        }
    }
    keys
}

#[test]
fn every_locale_parses_without_errors_or_duplicates() {
    for lang in Lang::ALL {
        let resource = FluentResource::try_new(lang.source().to_owned());
        assert!(resource.is_ok(), "{}: {:?}", lang.code(), resource.err().map(|(_, e)| e));
        let mut bundle = FluentBundle::<FluentResource>::new(vec![lang.langid()]);
        assert!(bundle.add_resource(resource.unwrap()).is_ok(), "{}: duplicate ids", lang.code());
    }
}

#[test]
fn translations_define_exactly_the_english_keys() {
    let english = ids(Lang::En.source());
    assert!(english.len() > 40, "id scan found only {english:?}");
    for lang in [Lang::Ru, Lang::ZhCn] {
        let theirs = ids(lang.source());
        let missing: Vec<_> = english.difference(&theirs).collect();
        let extra: Vec<_> = theirs.difference(&english).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{}: missing {missing:?}, extra {extra:?}",
            lang.code()
        );
    }
}

#[test]
fn every_key_used_by_the_code_exists_in_english() {
    let english = ids(Lang::En.source());
    let mut used: Vec<&str> = keys_used(include_str!("../app.rs"));
    used.extend(keys_used(include_str!("../main.rs")));
    used.extend(keys_used(include_str!("../inspector.rs")));
    used.extend(keys_used(include_str!("../inspector/panel.rs")));
    used.extend(keys_used(include_str!("../learn.rs")));
    used.extend(keys_used(include_str!("../structure.rs")));
    used.extend(keys_used(include_str!("../views.rs")));
    used.extend(Role::ALL.iter().flat_map(|r| <[&str; 2]>::from(r.message_keys())));
    used.extend(Role::ALL.iter().flat_map(|r| <[&str; 2]>::from(role_keys(*r))));
    used.extend(Presentation::ALL.map(Presentation::message_key));
    used.extend(Explanation::ALL.iter().flat_map(|e| [e.title_key(), e.body_key()]));
    // The route stores its keys in a const table, so `keys_used` cannot see them.
    used.extend(STEPS.iter().flat_map(|(title, body)| [*title, *body]));
    used.extend(
        [Basis::Confirmed, Basis::Inferred, Basis::Unknown]
            .iter()
            .flat_map(|b| <[&str; 2]>::from(b.message_keys())),
    );
    used.extend(Group::ALL.iter().flat_map(|g| <[&str; 2]>::from(group_keys(*g))));
    used.extend(Group::ALL.map(Group::message_key));
    used.extend(DIM_KEYS);
    // Dimension chips pick their hint through a helper, so the literals are listed here.
    used.extend(["dim-basis-confirmed-hint", "dim-basis-inferred-hint", "dim-basis-unknown-hint"]);
    used.extend([Mode::Simple, Mode::Expert].map(Mode::message_key));
    used.extend(["size-bytes", "size-kib", "size-mib", "size-gib"]);
    // Statistic rows look up "<key>-hint" for their tooltip.
    let stats =
        ["stat-count", "stat-min", "stat-max", "stat-mean", "stat-std", "stat-zeros", "stat-nan", "stat-inf"];
    let hints: Vec<String> = stats.iter().map(|s| format!("{s}-hint")).collect();
    used.extend(stats);
    used.extend(hints.iter().map(String::as_str));
    used.extend(Kind::ALL.iter().flat_map(|k| <[&str; 2]>::from(k.message_keys())));
    used.extend(FailureKind::ALL.iter().map(|k| k.message_key()));
    used.extend(ConfigProblem::ALL.iter().map(|p| p.message_key()));
    used.extend(["count-thousand", "count-million", "count-billion", "number-decimal-separator"]);
    assert!(used.len() > 30, "key scan found only {used:?}");
    let unknown: Vec<_> = used.iter().filter(|k| !english.contains(*k)).collect();
    assert!(unknown.is_empty(), "keys used in code but missing in en.ftl: {unknown:?}");
}

#[test]
fn missing_message_falls_back_to_english_then_to_the_key() {
    let t = I18n::from_sources(Lang::Ru, "a = Привет", "a = Hello\nb = Only English");
    assert_eq!(t.t("a"), "Привет");
    assert_eq!(t.t("b"), "Only English");
    // Nowhere defined: the key itself shows up, so a gap is visible instead of blank text.
    assert_eq!(t.t("no-such-key"), "no-such-key");
}

#[test]
fn broken_translation_falls_back_to_english() {
    // The Russian text uses an argument the code does not pass.
    let t = I18n::from_sources(Lang::Ru, "hi = Привет, { $nmae }", "hi = Hi, { $name }");
    assert_eq!(t.ta("hi", &[("name", "Ann".into())]), "Hi, Ann");
}

#[test]
fn plural_forms_follow_each_language() {
    let block = |lang: Lang, n: usize| {
        I18n::new(lang).ta("stack-blocks", &[("name", "m.layers".into()), ("count", n.into())])
    };
    assert_eq!(block(Lang::En, 1), "m.layers — 1 block");
    assert_eq!(block(Lang::En, 35), "m.layers — 35 blocks");
    assert_eq!(block(Lang::Ru, 1), "m.layers — 1 блок");
    assert_eq!(block(Lang::Ru, 3), "m.layers — 3 блока");
    assert_eq!(block(Lang::Ru, 12), "m.layers — 12 блоков");
    assert_eq!(block(Lang::Ru, 21), "m.layers — 21 блок");
    assert_eq!(block(Lang::ZhCn, 16), "m.layers — 16 个块");
    // No Unicode isolation marks around arguments: egui would draw them as boxes.
    assert!(!block(Lang::En, 2).contains(['\u{2068}', '\u{2069}']));
}

#[test]
fn counts_are_localized_and_round_into_the_next_unit() {
    let (en, ru, zh) = (I18n::new(Lang::En), I18n::new(Lang::Ru), I18n::new(Lang::ZhCn));
    assert_eq!(en.count(5_123_178_979), "5.12 B");
    assert_eq!(ru.count(5_123_178_979), "5,12 млрд");
    assert_eq!(zh.count(5_123_178_979), "5.12 十亿");
    assert_eq!(en.count(9_970_624), "10.0 M");
    assert_eq!(en.count(1_536), "1.5 K");
    assert_eq!(en.count(24), "24");
    assert_eq!(en.count(999_949), "999.9 K");
    assert_eq!(en.count(999_960), "1.0 M");
    assert_eq!(en.count(999_949_999), "999.9 M");
    assert_eq!(en.count(999_960_000), "1.00 B");
}

#[test]
fn os_locale_maps_to_supported_languages_only() {
    assert_eq!(Lang::from_locale("ru-RU"), Some(Lang::Ru));
    assert_eq!(Lang::from_locale("ru_RU.UTF-8"), Some(Lang::Ru));
    assert_eq!(Lang::from_locale("en-GB"), Some(Lang::En));
    assert_eq!(Lang::from_locale("zh-CN"), Some(Lang::ZhCn));
    assert_eq!(Lang::from_locale("zh"), Some(Lang::ZhCn));
    assert_eq!(Lang::from_locale("zh-Hans-TW"), Some(Lang::ZhCn));
    // Traditional Chinese and other languages are not supported.
    assert_eq!(Lang::from_locale("zh-TW"), None);
    assert_eq!(Lang::from_locale("zh-Hant"), None);
    assert_eq!(Lang::from_locale("de-DE"), None);
    assert_eq!(Lang::from_locale(""), None);
}

#[test]
fn initial_language_prefers_saved_then_os_then_english() {
    assert_eq!(initial_lang(Some("zh-CN"), Some("ru-RU")), Lang::ZhCn);
    // A corrupt saved value is ignored rather than trusted.
    assert_eq!(initial_lang(Some("klingon"), Some("ru-RU")), Lang::Ru);
    assert_eq!(initial_lang(None, Some("de-DE")), Lang::En);
    assert_eq!(initial_lang(None, None), Lang::En);
    for lang in Lang::ALL {
        assert_eq!(Lang::from_code(lang.code()), Some(lang), "storage code round-trips");
    }
}

#[test]
fn switching_language_applies_immediately() {
    let mut t = I18n::new(Lang::En);
    assert_eq!(t.t("toolbar-open-file"), "Open file…");
    t.set_lang(Lang::ZhCn);
    assert_eq!(t.lang(), Lang::ZhCn);
    assert_eq!(t.t("toolbar-open-file"), "打开文件…");
    t.set_lang(Lang::Ru);
    assert_eq!(t.t("toolbar-open-file"), "Открыть файл…");
}
