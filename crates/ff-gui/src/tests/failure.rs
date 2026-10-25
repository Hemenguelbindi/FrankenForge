//! Tests for module `failure`.

use std::io;
use std::path::PathBuf;

use super::*;

fn io_error(kind: ErrorKind) -> Error {
    Error::Io { path: PathBuf::from("m.safetensors"), source: io::Error::from(kind) }
}

#[test]
fn core_errors_map_to_explanations_and_keep_the_original_text() {
    let cases = [
        (io_error(ErrorKind::NotFound), FailureKind::NotFound),
        (io_error(ErrorKind::PermissionDenied), FailureKind::PermissionDenied),
        (io_error(ErrorKind::UnexpectedEof), FailureKind::Io),
        (
            Error::Format { path: "m".into(), reason: "header length 9 out of range".into() },
            FailureKind::Format,
        ),
        (Error::Incompatible("shape [2] vs [3]".into()), FailureKind::Incompatible),
        (Error::UnsupportedConfig("per-layer `layer_types`".into()), FailureKind::UnsupportedConfig),
        (Error::Recipe("step 1: no such source".into()), FailureKind::Recipe),
        (Error::Verify("w: content hash mismatch".into()), FailureKind::Verify),
        (Error::TensorNotFound("w".into()), FailureKind::Unknown),
    ];
    for (error, kind) in cases {
        let failure = Failure::from_error(&error);
        assert_eq!(failure.kind, kind, "{error}");
        // Never rewritten: the technical details are exactly what the core said.
        assert_eq!(failure.detail, error.to_string());
    }
    assert_eq!(Failure::crashed().kind, FailureKind::Crashed);
}
