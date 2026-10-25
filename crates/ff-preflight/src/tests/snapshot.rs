//! Tests for module `snapshot`: picking the volume of a path and an unknown GPU.

use super::*;

fn snap(disks: Known<Vec<DiskInfo>>, gpus: Known<Vec<GpuInfo>>) -> HardwareSnapshot {
    HardwareSnapshot {
        taken_at_unix: 0,
        cpu: Known::unknown("test"),
        logical_cores: Known::unknown("test"),
        ram_total: Known::unknown("test"),
        ram_available: Known::unknown("test"),
        ram_source: "test".into(),
        gpus,
        gpu_source: "test".into(),
        disks,
        disk_source: "test".into(),
    }
}

#[test]
fn the_longest_mount_prefix_wins() {
    let root = std::env::temp_dir();
    let inner = root.join("ff-mount");
    let disks = vec![
        DiskInfo { mount: root.clone(), total: 100, available: 10 },
        DiskInfo { mount: inner.clone(), total: 100, available: 77 },
    ];
    let s = snap(Known::Known(disks), Known::Known(vec![]));
    assert_eq!(s.disk_for(&inner.join("out")).get().unwrap().available, 77);
    assert_eq!(s.disk_for(&root.join("other")).get().unwrap().available, 10);
    // "ff-mount-2" shares a string prefix with "ff-mount" but is not inside it.
    assert_eq!(s.disk_for(&root.join("ff-mount-2")).get().unwrap().available, 10);
}

#[test]
fn unread_disks_and_missing_gpus_are_unknown() {
    let s = snap(Known::unknown("not read"), Known::Known(vec![]));
    assert!(matches!(s.disk_for(Path::new("x")), Known::Unknown(_)));
    assert!(matches!(s.gpu(), Known::Unknown(why) if why.contains("no GPU listed")));
}
