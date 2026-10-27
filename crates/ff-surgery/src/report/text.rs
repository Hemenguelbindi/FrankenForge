//! Text and Markdown views of a [`Report`]. Runs of consecutive blocks with the same status and
//! the same source (shifted by one per block) fold into one row, e.g. `0-14  recipient  unchanged`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::{BlockReport, Delta, FileCheck, Origin, Report, Stack, Status};

/// One or more consecutive blocks shown as one line.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub first: usize,
    pub last: usize,
    /// `alias` for unchanged blocks, otherwise `alias:index` or `alias:first-last`.
    pub source: String,
    /// Status of the first block; folded blocks have the same kind.
    pub status: Status,
    pub tensors: usize,
    /// Only rows with a computed or unknown delta; such rows never fold.
    pub delta: Option<Delta>,
}

impl Row {
    pub fn range(&self) -> String {
        if self.first == self.last { self.first.to_string() } else { format!("{}-{}", self.first, self.last) }
    }
}

impl Stack {
    pub fn rows(&self) -> Vec<Row> {
        let mut runs: Vec<(&BlockReport, usize)> = Vec::new();
        for b in &self.blocks {
            match runs.last_mut() {
                Some((first, len)) if continues(first, *len, b) => *len += 1,
                _ => runs.push((b, 1)),
            }
        }
        runs.into_iter()
            .map(|(b, len)| Row {
                first: b.index,
                last: b.index + len - 1,
                source: source_text(b, len),
                status: b.status.clone(),
                tensors: b.tensors,
                delta: b.delta.clone().filter(|d| *d != Delta::NotComputed),
            })
            .collect()
    }
}

/// Whether `next` extends the run of `len` blocks starting at `first`.
fn continues(first: &BlockReport, len: usize, next: &BlockReport) -> bool {
    let foldable = |b: &BlockReport| matches!(b.delta, None | Some(Delta::NotComputed));
    next.index == first.index + len
        && foldable(first)
        && foldable(next)
        && first.tensors == next.tensors
        && std::mem::discriminant(&first.status) == std::mem::discriminant(&next.status)
        && !matches!(first.status, Status::PartiallyReplaced { .. })
        && first.sources.len() == next.sources.len()
        && first
            .sources
            .iter()
            .zip(&next.sources)
            .all(|(a, b)| a.alias == b.alias && a.index.zip(b.index).is_some_and(|(x, y)| x + len == y))
}

fn source_text(b: &BlockReport, len: usize) -> String {
    if b.sources.is_empty() {
        return "generated".into();
    }
    let parts: Vec<String> = b
        .sources
        .iter()
        .map(|s| match (s.index, &b.status) {
            (None, _) | (_, Status::Unchanged) => s.alias.clone(),
            (Some(i), _) if len > 1 => format!("{}:{i}-{}", s.alias, i + len - 1),
            (Some(i), _) => format!("{}:{i}", s.alias),
        })
        .collect();
    parts.join(" + ")
}

fn status_text(row: &Row) -> String {
    match &row.status {
        Status::Unchanged => "unchanged".into(),
        Status::Shifted { .. } => "shifted".into(),
        Status::Replaced { .. } => format!("replaced ({0}/{0} tensors)", row.tensors),
        Status::PartiallyReplaced { replaced, total } => {
            format!("partially replaced ({replaced}/{total} tensors)")
        }
        Status::Added => "added".into(),
        Status::Generated => "generated (experimental)".into(),
    }
}

fn delta_text(d: &Delta) -> String {
    match d {
        Delta::NotComputed => "not computed".into(),
        Delta::Unknown { reason } => format!("unknown ({reason})"),
        Delta::Relative { value } => format!("{:.2}%", value * 100.0),
    }
}

impl Report {
    pub fn text(&self) -> String {
        let mut s = String::new();
        for stack in &self.stacks {
            let _ = writeln!(s, "{}: {} -> {} blocks", stack.name, stack.blocks_before, stack.blocks_after);
            let rows = stack.rows();
            let w_range = rows.iter().map(|r| r.range().len()).max().unwrap_or(0);
            let w_source = rows.iter().map(|r| r.source.len()).max().unwrap_or(0);
            for r in &rows {
                let mut line =
                    format!("  {:<w_range$}  {:<w_source$}  {}", r.range(), r.source, status_text(r));
                if let Some(d) = &r.delta {
                    let _ = write!(line, "   weight delta: {}", delta_text(d));
                }
                let _ = writeln!(s, "{line}");
            }
        }
        if !self.removed_blocks.is_empty() {
            let names: Vec<String> =
                self.removed_blocks.iter().map(|b| format!("{}.{}", b.stack, b.index)).collect();
            let _ = writeln!(s, "removed blocks: {} <- {}", names.join(", "), self.base);
        }
        if !self.removed_tensors.is_empty() {
            let _ = writeln!(s, "removed tensors: {} <- {}", self.removed_tensors.join(", "), self.base);
        }
        self.outside_text(&mut s);
        match (self.config_present, self.config_changes.is_empty()) {
            (false, _) => s.push_str("config.json: none in base\n"),
            (true, true) => s.push_str("config.json: unchanged\n"),
            (true, false) => self.config_changes.iter().for_each(|c| {
                let _ = writeln!(s, "{c}");
            }),
        }
        for w in &self.warnings {
            let _ = writeln!(s, "warning: {w}");
        }
        if let Some(e) = &self.experimental {
            s.push_str("EXPERIMENTAL, not a working model:\n");
            e.text().lines().for_each(|l| {
                let _ = writeln!(s, "  {l}");
            });
        }
        s.push_str(match self.file {
            FileCheck::NotWritten => "file: not written (plan only)\n",
            FileCheck::Verified => "file: all tensors re-read, hashes match\n",
        });
        s.push_str("model behavior: NOT EVALUATED\n");
        s
    }

    /// Unchanged tensors are grouped by source; everything else gets its own line.
    fn outside_text(&self, s: &mut String) {
        let mut unchanged: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut other = Vec::new();
        for o in &self.outside {
            match &o.origin {
                Origin::Source { alias, .. } if o.unchanged => {
                    unchanged.entry(alias).or_default().push(&o.name)
                }
                Origin::Source { alias, name } => {
                    let what = match &o.delta {
                        Some(d) if *d == Delta::NotComputed => "replaced".to_owned(),
                        Some(d) => format!("replaced, weight delta: {}", delta_text(d)),
                        None => "added".to_owned(),
                    };
                    other.push(format!("{} <- {alias}:{name}, {what}", o.name));
                }
                Origin::Generated => other.push(format!("{} <- generated (experimental)", o.name)),
            }
        }
        for (alias, names) in unchanged {
            let _ = writeln!(s, "outside blocks: {} <- {alias}, unchanged", names.join(", "));
        }
        for line in other {
            let _ = writeln!(s, "outside blocks: {line}");
        }
    }

    pub fn markdown(&self) -> String {
        format!(
            "# FrankenForge build report\n\nBehavior of the result was not evaluated: nothing was run.\n\n```text\n{}```\n",
            self.text()
        )
    }
}

#[cfg(test)]
#[path = "tests/text.rs"]
mod tests;
