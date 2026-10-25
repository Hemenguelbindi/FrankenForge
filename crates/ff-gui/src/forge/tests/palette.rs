//! Tests for module `forge::palette`: both themes keep text and role colors readable.

use super::*;

/// WCAG contrast ratio of two opaque colors.
fn contrast(a: Color32, b: Color32) -> f32 {
    let lum = |c: Color32| {
        let ch = |v: u8| {
            let s = f32::from(v) / 255.0;
            if s <= 0.03928 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
    };
    let (x, y) = (lum(a), lum(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

#[test]
fn each_theme_gets_its_own_palette() {
    assert_eq!(Palette::of(&Visuals::dark()), DARK);
    assert_eq!(Palette::of(&Visuals::light()), LIGHT);
}

#[test]
fn text_is_readable_on_every_surface_in_both_themes() {
    for p in [DARK, LIGHT] {
        for bg in [p.surface, p.card, p.well] {
            assert!(contrast(p.text, bg) >= 7.0, "text on {bg:?}: {}", contrast(p.text, bg));
            assert!(contrast(p.muted, bg) >= 4.5, "muted on {bg:?}: {}", contrast(p.muted, bg));
        }
    }
}

#[test]
fn role_colors_stand_out_from_cards_and_their_tints() {
    for p in [DARK, LIGHT] {
        for role in [p.base, p.donor, p.seam, p.ok, p.error] {
            assert!(contrast(role, p.card) >= 3.0, "{role:?} on card: {}", contrast(role, p.card));
            assert!(contrast(role, p.tint(role)) >= 3.0, "{role:?} on its tint");
        }
    }
}

#[test]
fn base_and_donor_are_not_the_same_hue_family() {
    for p in [DARK, LIGHT] {
        // Cold Base: blue dominates red; warm Donor: red dominates blue.
        assert!(p.base.b() > p.base.r() && p.donor.r() > p.donor.b());
    }
}
