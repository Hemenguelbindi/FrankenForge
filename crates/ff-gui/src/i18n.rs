//! Localization with Project Fluent. Translations live in `locales/*.ftl` and are embedded at
//! build time; English is the fallback for any message a locale lacks or fails to format.

use fluent_bundle::{FluentArgs, FluentBundle, FluentResource, FluentValue};
use unic_langid::LanguageIdentifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ru,
    ZhCn,
}

impl Lang {
    pub const ALL: [Lang; 3] = [Lang::En, Lang::Ru, Lang::ZhCn];

    /// Stable id used for storage and `--lang`.
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ru => "ru",
            Lang::ZhCn => "zh-CN",
        }
    }

    /// The language's own name, shown in the picker whatever the interface language is.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Ru => "Русский",
            Lang::ZhCn => "简体中文",
        }
    }

    pub fn from_code(code: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.code().eq_ignore_ascii_case(code.trim()))
    }

    /// Maps an OS locale (`ru-RU`, `zh_CN.UTF-8`, `zh-Hans-TW`) to a supported language.
    /// Traditional Chinese is not supported, so it maps to `None` rather than to Simplified.
    pub fn from_locale(tag: &str) -> Option<Lang> {
        let tag = tag.split(['.', '@']).next()?.replace('_', "-").to_ascii_lowercase();
        let mut parts = tag.split('-');
        match parts.next()? {
            "en" => Some(Lang::En),
            "ru" => Some(Lang::Ru),
            "zh" => {
                let rest: Vec<&str> = parts.collect();
                let simplified =
                    rest.contains(&"hans") || !rest.iter().any(|p| matches!(*p, "hant" | "tw" | "hk" | "mo"));
                simplified.then_some(Lang::ZhCn)
            }
            _ => None,
        }
    }

    pub(crate) fn source(self) -> &'static str {
        match self {
            Lang::En => include_str!("../locales/en.ftl"),
            Lang::Ru => include_str!("../locales/ru.ftl"),
            Lang::ZhCn => include_str!("../locales/zh-CN.ftl"),
        }
    }

    fn langid(self) -> LanguageIdentifier {
        self.code().parse().expect("the codes above are valid BCP 47 tags")
    }
}

/// A saved choice wins, then the OS language, then English.
pub fn initial_lang(saved: Option<&str>, os_locale: Option<&str>) -> Lang {
    saved.and_then(Lang::from_code).or_else(|| os_locale.and_then(Lang::from_locale)).unwrap_or(Lang::En)
}

pub fn os_locale() -> Option<String> {
    sys_locale::get_locale()
}

pub type Args = [(&'static str, FluentValue<'static>)];

pub struct I18n {
    lang: Lang,
    current: FluentBundle<FluentResource>,
    english: FluentBundle<FluentResource>,
}

impl I18n {
    pub fn new(lang: Lang) -> Self {
        Self::from_sources(lang, lang.source(), Lang::En.source())
    }

    pub(crate) fn from_sources(lang: Lang, current: &str, english: &str) -> Self {
        Self { lang, current: bundle(lang, current), english: bundle(Lang::En, english) }
    }

    pub fn lang(&self) -> Lang {
        self.lang
    }

    pub fn set_lang(&mut self, lang: Lang) {
        if lang != self.lang {
            *self = Self::new(lang);
        }
    }

    pub fn t(&self, key: &str) -> String {
        self.format(key, None)
    }

    pub fn ta(&self, key: &str, args: &Args) -> String {
        let mut fluent = FluentArgs::new();
        for (name, value) in args {
            fluent.set(*name, value.clone());
        }
        self.format(key, Some(&fluent))
    }

    /// Current locale, then English, then the key itself so a gap is visible instead of blank.
    fn format(&self, key: &str, args: Option<&FluentArgs>) -> String {
        lookup(&self.current, key, args)
            .or_else(|| lookup(&self.english, key, args))
            .unwrap_or_else(|| key.to_owned())
    }

    /// `5123178979` -> `5.12 B` in English; the unit and decimal separator come from the locale
    /// file. Small counts stay exact. Thresholds
    /// sit where rounding would otherwise print `1000.0 K` instead of `1.0 M`.
    pub fn count(&self, n: usize) -> String {
        let x = n as f64;
        let (value, key) = match n {
            999_950_000.. => (format!("{:.2}", x / 1e9), "count-billion"),
            999_950.. => (format!("{:.1}", x / 1e6), "count-million"),
            1_000.. => (format!("{:.1}", x / 1e3), "count-thousand"),
            _ => return n.to_string(),
        };
        let value = value.replace('.', &self.t("number-decimal-separator"));
        self.ta(key, &[("value", value.into())])
    }
}

impl I18n {
    /// Byte size in binary units (`4.48 GiB`), localized unit names and decimal separator.
    pub fn bytes(&self, n: u64) -> String {
        const UNITS: [(f64, &str); 3] =
            [(1_073_741_824.0, "size-gib"), (1_048_576.0, "size-mib"), (1024.0, "size-kib")];
        let x = n as f64;
        let Some((unit, key)) = UNITS.into_iter().find(|(unit, _)| x >= *unit) else {
            return self.ta("size-bytes", &[("value", n.into())]);
        };
        let value = format!("{:.2}", x / unit).replace('.', &self.t("number-decimal-separator"));
        self.ta(key, &[("value", value.into())])
    }
}

fn bundle(lang: Lang, source: &str) -> FluentBundle<FluentResource> {
    // Bundled files are checked by tests; at runtime keep whatever parsed rather than fail.
    let resource = FluentResource::try_new(source.to_owned()).unwrap_or_else(|(partial, _)| partial);
    let mut bundle = FluentBundle::new(vec![lang.langid()]);
    // Unicode isolation marks around arguments would render as boxes in egui.
    bundle.set_use_isolating(false);
    // Only duplicate ids fail here, and tests rule them out; the first definition is kept.
    let _ = bundle.add_resource(resource);
    bundle
}

/// `None` when the message is missing or a translation could not be formatted (e.g. it uses an
/// argument the code does not pass), so the caller falls back to English.
fn lookup(bundle: &FluentBundle<FluentResource>, key: &str, args: Option<&FluentArgs>) -> Option<String> {
    let pattern = bundle.get_message(key)?.value()?;
    let mut errors = Vec::new();
    let text = bundle.format_pattern(pattern, args, &mut errors);
    errors.is_empty().then(|| text.into_owned())
}

#[cfg(test)]
#[path = "tests/i18n.rs"]
mod tests;
