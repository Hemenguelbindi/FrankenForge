//! Forge colors by meaning, one set per theme. Drawing code takes every color from here, so the
//! light theme and later tuning never mean hunting literals through the views. The app theme and
//! the Inspector are not affected.

use eframe::egui::{Color32, Visuals};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    /// The Base model: cold cyan/blue.
    pub base: Color32,
    /// The Donor model: warm amber/copper.
    pub donor: Color32,
    /// Where a Donor block joins the Base.
    pub seam: Color32,
    pub ok: Color32,
    pub error: Color32,
    /// Forge page background.
    pub surface: Color32,
    /// Cards and panels on the surface.
    pub card: Color32,
    /// Empty cells, bars and other quiet fills inside a card.
    pub well: Color32,
    pub border: Color32,
    pub text: Color32,
    pub muted: Color32,
}

const DARK: Palette = Palette {
    base: Color32::from_rgb(56, 189, 248),
    donor: Color32::from_rgb(234, 150, 76),
    seam: Color32::from_rgb(172, 148, 250),
    ok: Color32::from_rgb(52, 211, 153),
    error: Color32::from_rgb(248, 113, 113),
    surface: Color32::from_rgb(15, 19, 26),
    card: Color32::from_rgb(24, 30, 40),
    well: Color32::from_rgb(33, 40, 52),
    border: Color32::from_rgb(52, 62, 79),
    text: Color32::from_rgb(228, 233, 241),
    muted: Color32::from_rgb(150, 162, 181),
};

const LIGHT: Palette = Palette {
    base: Color32::from_rgb(3, 105, 161),
    donor: Color32::from_rgb(166, 82, 18),
    seam: Color32::from_rgb(109, 40, 217),
    ok: Color32::from_rgb(4, 120, 87),
    error: Color32::from_rgb(190, 30, 30),
    surface: Color32::from_rgb(238, 242, 247),
    card: Color32::from_rgb(255, 255, 255),
    well: Color32::from_rgb(229, 234, 241),
    border: Color32::from_rgb(196, 206, 220),
    text: Color32::from_rgb(15, 23, 42),
    muted: Color32::from_rgb(80, 94, 114),
};

impl Palette {
    pub fn of(visuals: &Visuals) -> Self {
        if visuals.dark_mode { DARK } else { LIGHT }
    }

    /// A quiet fill of `color` for backgrounds behind text in that color.
    pub fn tint(self, color: Color32) -> Color32 {
        let k = if self.card == DARK.card { 0.18 } else { 0.12 };
        lerp(self.card, color, k)
    }
}

fn lerp(a: Color32, b: Color32, k: f32) -> Color32 {
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * k).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

#[cfg(test)]
#[path = "tests/palette.rs"]
mod tests;
