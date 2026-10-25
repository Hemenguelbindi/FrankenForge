//! Tests for module `fonts`: the bundled font must draw every character the interface can show.

use crate::i18n::Lang;

use super::NOTO_SANS_SC;

#[test]
fn bundled_font_covers_every_character_of_every_locale() {
    let face = ttf_parser::Face::parse(NOTO_SANS_SC, 0).expect("bundled font parses");
    let mut text: String = Lang::ALL.iter().map(|l| l.source()).collect();
    text.extend(Lang::ALL.iter().map(|l| l.native_name()));
    let mut missing: Vec<char> =
        text.chars().filter(|c| !c.is_control() && *c != ' ' && face.glyph_index(*c).is_none()).collect();
    missing.sort_unstable();
    missing.dedup();
    assert!(missing.is_empty(), "no glyph for {missing:?}");
    // Spot checks for each script, in case the locale files ever shrink.
    for c in ['A', 'Ж', 'ё', '中', '张', '量', '…', '—', '·'] {
        assert!(face.glyph_index(c).is_some(), "{c}");
    }
}
