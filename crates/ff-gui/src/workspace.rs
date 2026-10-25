//! The two top-level sections: reading one checkpoint (Inspector), or building a new one from two
//! (Forge). Switching is the only Forge integration the app shell needs.

use eframe::egui::{self, RichText};

use crate::i18n::I18n;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Workspace {
    #[default]
    Inspector,
    Forge,
}

impl Workspace {
    pub const ALL: [Workspace; 2] = [Workspace::Inspector, Workspace::Forge];

    pub fn message_key(self) -> &'static str {
        match self {
            Workspace::Inspector => "workspace-inspector",
            Workspace::Forge => "workspace-forge",
        }
    }
}

/// Toolbar switch; returns true when the user just left the Inspector.
pub fn switcher(ui: &mut egui::Ui, t: &I18n, current: &mut Workspace) -> bool {
    let before = *current;
    for w in Workspace::ALL {
        if ui.selectable_label(*current == w, RichText::new(t.t(w.message_key())).strong()).clicked() {
            *current = w;
        }
    }
    before == Workspace::Inspector && *current != Workspace::Inspector
}
