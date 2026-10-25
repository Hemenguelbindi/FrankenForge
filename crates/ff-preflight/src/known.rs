//! A value that may be unknown, and a byte range for honest estimates.

/// `Unknown` carries the reason; it is never read as zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Known<T> {
    Known(T),
    Unknown(String),
}

impl<T> Known<T> {
    pub fn unknown(why: impl Into<String>) -> Self {
        Known::Unknown(why.into())
    }

    pub fn get(&self) -> Option<&T> {
        match self {
            Known::Known(v) => Some(v),
            Known::Unknown(_) => None,
        }
    }

    pub fn as_ref(&self) -> Known<&T> {
        match self {
            Known::Known(v) => Known::Known(v),
            Known::Unknown(why) => Known::Unknown(why.clone()),
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Known<U> {
        match self {
            Known::Known(v) => Known::Known(f(v)),
            Known::Unknown(why) => Known::Unknown(why),
        }
    }
}

/// An inclusive byte range `low..=high`; an exact size has `low == high`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub low: u64,
    pub high: u64,
}

impl Span {
    pub fn exact(bytes: u64) -> Self {
        Self { low: bytes, high: bytes }
    }

    pub fn between(a: u64, b: u64) -> Self {
        Self { low: a.min(b), high: a.max(b) }
    }

    pub fn plus(self, other: Span) -> Self {
        Self { low: self.low.saturating_add(other.low), high: self.high.saturating_add(other.high) }
    }
}

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;
