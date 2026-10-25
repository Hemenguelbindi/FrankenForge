//! Top-level presentation chosen in the toolbar: a guided tour, a plain-language model map, or
//! the raw technical inspector. The choice is saved with the other UI settings.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Presentation {
    /// Default for a reader who does not know ML yet.
    #[default]
    Learn,
    Structure,
    Technical,
}

impl Presentation {
    pub const ALL: [Presentation; 3] =
        [Presentation::Learn, Presentation::Structure, Presentation::Technical];

    pub fn code(self) -> &'static str {
        match self {
            Presentation::Learn => "learn",
            Presentation::Structure => "structure",
            Presentation::Technical => "technical",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        Presentation::ALL.into_iter().find(|p| p.code().eq_ignore_ascii_case(code.trim()))
    }

    pub fn message_key(self) -> &'static str {
        match self {
            Presentation::Learn => "presentation-learn",
            Presentation::Structure => "presentation-structure",
            Presentation::Technical => "presentation-technical",
        }
    }
}

/// A saved choice wins; a corrupt or missing value falls back to the guided tour.
pub fn initial_presentation(saved: Option<&str>) -> Presentation {
    saved.and_then(Presentation::from_code).unwrap_or_default()
}

#[cfg(test)]
#[path = "tests/presentation.rs"]
mod tests;
