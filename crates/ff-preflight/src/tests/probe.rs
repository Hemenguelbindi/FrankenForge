//! Tests for module `probe`: nvidia-smi output parsing, and an ignored read of this machine.

use super::*;

#[test]
fn parses_nvidia_smi_csv_in_mib() {
    let gpus = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 10098\n\n").unwrap();
    assert_eq!(
        gpus,
        [GpuInfo { name: "NVIDIA GeForce RTX 3060".into(), vram_total: 12288 * MIB, vram_free: 10098 * MIB }]
    );
    // A comma inside the name stays part of the name.
    let gpus = parse_nvidia_smi("Lab GPU, rev 2, 8192, 100").unwrap();
    assert_eq!(gpus[0].name, "Lab GPU, rev 2");
    assert!(parse_nvidia_smi("").unwrap().is_empty());
}

#[test]
fn malformed_output_is_an_error_not_zero() {
    assert!(parse_nvidia_smi("GPU, [N/A], [N/A]").is_err());
    assert!(parse_nvidia_smi("just text").is_err());
}

/// Prints what this machine reports: `cargo test -p ff-preflight probe -- --ignored --nocapture`.
#[test]
#[ignore = "reads the real machine"]
fn reads_this_machine() {
    let s = probe();
    println!("{s:#?}");
    assert!(s.taken_at_unix > 0);
}
