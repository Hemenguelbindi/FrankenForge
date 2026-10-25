//! Why a request (opening, analysis, planning) failed, in two parts: a kind that selects a localized explanation, and the
//! original error text, kept unchanged for the "technical details" section.

use std::io::ErrorKind;

use ff_core::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    NotFound,
    PermissionDenied,
    Io,
    Format,
    /// Blocks or tensors of the two models do not fit (names, types or shapes).
    Incompatible,
    /// config.json cannot be updated safely, so the core refuses.
    UnsupportedConfig,
    /// The generated recipe could not be applied.
    Recipe,
    /// The written output did not match the plan.
    Verify,
    Crashed,
    /// Superseded by a newer request; never shown, the newer result replaces it.
    Cancelled,
    Unknown,
}

impl FailureKind {
    #[cfg(test)]
    pub const ALL: [FailureKind; 11] = [
        FailureKind::NotFound,
        FailureKind::PermissionDenied,
        FailureKind::Io,
        FailureKind::Format,
        FailureKind::Incompatible,
        FailureKind::UnsupportedConfig,
        FailureKind::Recipe,
        FailureKind::Verify,
        FailureKind::Crashed,
        FailureKind::Cancelled,
        FailureKind::Unknown,
    ];

    pub fn message_key(self) -> &'static str {
        match self {
            FailureKind::NotFound => "error-not-found",
            FailureKind::PermissionDenied => "error-permission",
            FailureKind::Io => "error-io",
            FailureKind::Format => "error-format",
            FailureKind::Incompatible => "error-incompatible",
            FailureKind::UnsupportedConfig => "error-unsupported-config",
            FailureKind::Recipe => "error-recipe",
            FailureKind::Verify => "error-verify",
            FailureKind::Crashed => "error-crashed",
            FailureKind::Cancelled => "error-cancelled",
            FailureKind::Unknown => "error-unknown",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Failure {
    pub kind: FailureKind,
    /// The error exactly as the core reported it.
    pub detail: String,
}

impl Failure {
    pub fn from_error(error: &Error) -> Self {
        let kind = match error {
            Error::Io { source, .. } => match source.kind() {
                ErrorKind::NotFound => FailureKind::NotFound,
                ErrorKind::PermissionDenied => FailureKind::PermissionDenied,
                _ => FailureKind::Io,
            },
            Error::Format { .. } => FailureKind::Format,
            Error::Incompatible(_) => FailureKind::Incompatible,
            Error::UnsupportedConfig(_) => FailureKind::UnsupportedConfig,
            Error::Recipe(_) => FailureKind::Recipe,
            Error::Verify(_) => FailureKind::Verify,
            _ => FailureKind::Unknown,
        };
        Self { kind, detail: error.to_string() }
    }

    pub fn crashed() -> Self {
        Self {
            kind: FailureKind::Crashed,
            detail: "the worker thread panicked before returning a result".into(),
        }
    }
}

#[cfg(test)]
#[path = "tests/failure.rs"]
mod tests;
