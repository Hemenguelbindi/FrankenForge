//! FrankenForge GUI: a read-only checkpoint inspector. Reads metadata only; never loads
//! weights and runs no inference.
//!
//! Usage: `ff-gui [--lang en|ru|zh-CN] [checkpoint]`.

// No console window behind the GUI in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod app;
mod components;
mod failure;
mod fonts;
mod i18n;
mod inspected;
mod inspector;
mod learn;
mod loader;
mod presentation;
mod semantics;
mod structure;
mod theme;
mod views;

use std::ffi::OsString;
use std::path::PathBuf;

use eframe::egui;

use crate::i18n::{I18n, Lang, initial_lang, os_locale};

/// `--lang <code>` overrides the saved language for this run; anything else is the checkpoint.
/// An unknown code is ignored, so the normal saved/OS/English choice applies.
fn parse_args(args: impl IntoIterator<Item = OsString>) -> (Option<PathBuf>, Option<Lang>) {
    let (mut path, mut lang) = (None, None);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--lang" {
            lang = args.next().and_then(|v| v.into_string().ok()).and_then(|v| Lang::from_code(&v));
        } else {
            path = Some(PathBuf::from(arg));
        }
    }
    (path, lang)
}

fn main() -> eframe::Result {
    let (path, forced) = parse_args(std::env::args_os().skip(1));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("FrankenForge")
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([640.0, 420.0]),
        ..Default::default()
    };
    let result = eframe::run_native(
        "FrankenForge",
        options,
        Box::new(move |cc| {
            fonts::install(&cc.egui_ctx);
            theme::install(&cc.egui_ctx);
            Ok(Box::new(app::App::new(cc, path, forced)))
        }),
    );
    // Release builds have no console, so a startup failure (no GPU, failed window) would be silent.
    // Saved settings are not readable without the window, so the OS language is used here.
    if let Err(e) = &result {
        let t = I18n::new(forced.unwrap_or_else(|| initial_lang(None, os_locale().as_deref())));
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("FrankenForge")
            .set_description(format!("{}\n\n{}:\n{e}", t.t("startup-failed"), t.t("error-details")))
            .show();
    }
    result
}

#[cfg(test)]
#[path = "tests/main.rs"]
mod tests;
