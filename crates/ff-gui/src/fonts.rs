//! Bundled Noto Sans SC (SIL OFL 1.1, see `assets/fonts/OFL.txt`) covers Latin, Cyrillic and
//! Simplified Chinese, so every locale renders the same on any machine, with or without system fonts.

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

pub const NOTO_SANS_SC: &[u8] = include_bytes!("../assets/fonts/NotoSansSC-Regular.otf");
const NAME: &str = "noto-sans-sc";

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(NAME.into(), Arc::new(FontData::from_static(NOTO_SANS_SC)));
    // First for UI text, so all three scripts share one typeface; a fallback for monospace, so
    // tensor names keep the code font while CJK in config.json still renders.
    fonts.families.entry(FontFamily::Proportional).or_default().insert(0, NAME.into());
    fonts.families.entry(FontFamily::Monospace).or_default().push(NAME.into());
    ctx.set_fonts(fonts);
}

#[cfg(test)]
#[path = "tests/fonts.rs"]
mod tests;
