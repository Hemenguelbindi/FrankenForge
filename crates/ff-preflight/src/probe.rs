//! Reads the machine. The only module with system access; call it off the UI thread, on demand,
//! never per frame. Every fact keeps its source; anything not read stays Unknown.

use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};

use crate::known::{Known, MIB};
use crate::snapshot::{DiskInfo, GpuInfo, HardwareSnapshot};

const SYSINFO: &str = "sysinfo 0.39";
const NVIDIA_SMI: &str = "nvidia-smi --query-gpu=name,memory.total,memory.free";

pub fn probe() -> HardwareSnapshot {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::nothing().with_ram())
            .with_cpu(CpuRefreshKind::nothing()),
    );
    let known_nonzero = |v: u64, what: &str| {
        if v > 0 { Known::Known(v) } else { Known::unknown(format!("{SYSINFO} reported 0 for {what}")) }
    };
    let cpu = sys.cpus().first().map(|c| c.brand().trim().to_owned()).filter(|b| !b.is_empty());
    HardwareSnapshot {
        taken_at_unix: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        cpu: cpu.map_or_else(|| Known::unknown(format!("{SYSINFO}: no CPU brand")), Known::Known),
        logical_cores: match sys.cpus().len() {
            0 => Known::unknown(format!("{SYSINFO}: no CPUs listed")),
            n => Known::Known(n),
        },
        ram_total: known_nonzero(sys.total_memory(), "total memory"),
        ram_available: known_nonzero(sys.available_memory(), "available memory"),
        ram_source: SYSINFO.into(),
        gpus: nvidia_smi(),
        gpu_source: NVIDIA_SMI.into(),
        disks: Known::Known(disks()),
        disk_source: SYSINFO.into(),
    }
}

fn disks() -> Vec<DiskInfo> {
    Disks::new_with_refreshed_list()
        .list()
        .iter()
        .map(|d| DiskInfo {
            mount: d.mount_point().to_owned(),
            total: d.total_space(),
            available: d.available_space(),
        })
        .collect()
}

/// Free space on the volume holding `path`, read now (for the check right before a write).
pub fn disk_available(path: &Path) -> Known<u64> {
    let snap = HardwareSnapshot {
        taken_at_unix: 0,
        cpu: Known::unknown("not read"),
        logical_cores: Known::unknown("not read"),
        ram_total: Known::unknown("not read"),
        ram_available: Known::unknown("not read"),
        ram_source: String::new(),
        gpus: Known::unknown("not read"),
        gpu_source: String::new(),
        disks: Known::Known(disks()),
        disk_source: SYSINFO.into(),
    };
    snap.disk_for(path).map(|d| d.available)
}

fn nvidia_smi() -> Known<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Err(e) if e.kind() == ErrorKind::NotFound => {
            Known::unknown("nvidia-smi not found; this does not prove there is no NVIDIA GPU")
        }
        Err(e) => Known::unknown(format!("nvidia-smi could not run: {e}")),
        Ok(o) if !o.status.success() => Known::unknown(format!(
            "nvidia-smi exited with {}: {}",
            o.status,
            String::from_utf8_lossy(&o.stderr).trim()
        )),
        Ok(o) => match parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout)) {
            Ok(gpus) => Known::Known(gpus),
            Err(why) => Known::Unknown(why),
        },
    }
}

/// `name, total MiB, free MiB` per line (`--format=csv,noheader,nounits`).
pub fn parse_nvidia_smi(stdout: &str) -> Result<Vec<GpuInfo>, String> {
    stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.rsplitn(3, ',').map(str::trim).collect();
            let [free, total, name] = f[..] else {
                return Err(format!("unexpected nvidia-smi line: {line}"));
            };
            let mib = |s: &str| {
                s.parse::<u64>().map(|v| v * MIB).map_err(|_| format!("not a number: {s:?} in {line}"))
            };
            Ok(GpuInfo { name: name.to_owned(), vram_total: mib(total)?, vram_free: mib(free)? })
        })
        .collect()
}

#[cfg(test)]
#[path = "tests/probe.rs"]
mod tests;
