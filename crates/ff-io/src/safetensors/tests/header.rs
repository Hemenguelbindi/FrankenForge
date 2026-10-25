//! Tests for module `header`.

use super::*;

#[test]
fn rejects_offsets_past_data() {
    let json = br#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    assert!(parse(json, 8).is_err());
    assert!(parse(json, 16).is_ok());
}

#[test]
fn rejects_size_mismatch() {
    let json = br#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,16]}}"#;
    assert!(parse(json, 16).is_err());
}
