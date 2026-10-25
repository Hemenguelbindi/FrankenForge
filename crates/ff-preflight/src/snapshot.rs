//! What the machine reported at one moment, with the source of each group of facts.

use std::path::{Path, PathBuf};

use crate::known::Known;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskInfo {
    pub mount: PathBuf,
    pub total: u64,
    pub available: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareSnapshot {
    /// Seconds since the Unix epoch when the facts were read.
    pub taken_at_unix: u64,
    pub cpu: Known<String>,
    pub logical_cores: Known<usize>,
    pub ram_total: Known<u64>,
    pub ram_available: Known<u64>,
    pub ram_source: String,
    pub gpus: Known<Vec<GpuInfo>>,
    pub gpu_source: String,
    pub disks: Known<Vec<DiskInfo>>,
    pub disk_source: String,
}

impl HardwareSnapshot {
    // shortcut: the first reported GPU is the one used; add a choice when a machine has several.
    pub fn gpu(&self) -> Known<&GpuInfo> {
        match &self.gpus {
            Known::Known(list) => match list.first() {
                Some(g) => Known::Known(g),
                None => Known::unknown(format!("{}: no GPU listed", self.gpu_source)),
            },
            Known::Unknown(why) => Known::Unknown(why.clone()),
        }
    }

    /// The volume holding `path`: the disk whose mount point is the longest prefix of it.
    pub fn disk_for(&self, path: &Path) -> Known<&DiskInfo> {
        let Known::Known(disks) = &self.disks else {
            return Known::unknown(format!("{}: disks not read", self.disk_source));
        };
        let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
        let key = |p: &Path| {
            if cfg!(windows) { p.to_string_lossy().to_lowercase() } else { p.to_string_lossy().into_owned() }
        };
        let target = key(&path);
        disks
            .iter()
            .filter(|d| Path::new(&target).starts_with(key(&d.mount)))
            .max_by_key(|d| d.mount.as_os_str().len())
            .map_or_else(|| Known::unknown(format!("no disk contains {}", path.display())), Known::Known)
    }
}

#[cfg(test)]
#[path = "tests/snapshot.rs"]
mod tests;
