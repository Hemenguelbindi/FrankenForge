//! Display model of the block schema: each model gets its own lane with its own length, links
//! join a Donor block to a Base block explicitly, and every block carries a transfer category.
//! Pure data and layout arithmetic; drawing lives in `schema_view`.

use std::collections::BTreeMap;
use std::ops::Range;

use super::analysis::{Analysis, BlockId, BlockStatus};

/// What moving a block would take. Only `Direct` is implemented; the others explain why a block
/// is out of reach and never enable Build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    Direct,
    /// The block has no partner at the same position; an explicit, checked mapping would be needed.
    NeedsMapping,
    /// Reserved for a future experimental mode; nothing infers it today.
    NeedsAdapter,
    Unsupported,
}

impl Transfer {
    pub const ALL: [Transfer; 4] =
        [Transfer::Direct, Transfer::NeedsMapping, Transfer::NeedsAdapter, Transfer::Unsupported];

    pub fn available(self) -> bool {
        self == Transfer::Direct
    }

    pub fn message_key(self) -> &'static str {
        match self {
            Transfer::Direct => "forge-transfer-direct",
            Transfer::NeedsMapping => "forge-transfer-mapping",
            Transfer::NeedsAdapter => "forge-transfer-adapter",
            Transfer::Unsupported => "forge-transfer-unsupported",
        }
    }

    /// A structural difference is not evidence that an adapter would work, so it stays Unsupported.
    pub fn of(status: &BlockStatus) -> Self {
        match status {
            BlockStatus::Compatible => Transfer::Direct,
            BlockStatus::OnlyInBase | BlockStatus::OnlyInDonor => Transfer::NeedsMapping,
            BlockStatus::Different(_) => Transfer::Unsupported,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Base,
    Donor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    Block {
        index: usize,
        transfer: Transfer,
    },
    /// A skipped index inside this model's own numbering.
    Gap {
        index: usize,
    },
}

/// One model's blocks; `cells[i]` is position `i`, and the lane ends at that model's last block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    pub side: Side,
    pub cells: Vec<Cell>,
}

/// Donor block `donor` replaces Base block `base` in one stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub stack: String,
    pub donor: usize,
    pub base: usize,
}

pub struct StackSchema {
    pub stack: String,
    pub base: Lane,
    pub donor: Lane,
    /// The core's description of each structural difference, by index.
    pub differences: BTreeMap<usize, Vec<String>>,
}

impl StackSchema {
    /// Positions to lay out: the longer of the two lanes.
    pub fn width(&self) -> usize {
        self.base.cells.len().max(self.donor.cells.len())
    }
}

pub fn stacks(a: &Analysis) -> Vec<StackSchema> {
    let mut by_stack: BTreeMap<&str, Vec<&super::analysis::BlockRow>> = BTreeMap::new();
    for row in &a.rows {
        by_stack.entry(&row.id.stack).or_default().push(row);
    }
    by_stack
        .into_iter()
        .map(|(stack, rows)| {
            let side = |skip: &BlockStatus| -> BTreeMap<usize, Transfer> {
                let present = rows.iter().filter(|r| r.status != *skip);
                present.map(|r| (r.id.index, Transfer::of(&r.status))).collect()
            };
            let differences = rows
                .iter()
                .filter_map(|r| match &r.status {
                    BlockStatus::Different(d) => Some((r.id.index, d.clone())),
                    _ => None,
                })
                .collect();
            StackSchema {
                stack: stack.to_owned(),
                base: lane(Side::Base, &side(&BlockStatus::OnlyInDonor)),
                donor: lane(Side::Donor, &side(&BlockStatus::OnlyInBase)),
                differences,
            }
        })
        .collect()
}

fn lane(side: Side, present: &BTreeMap<usize, Transfer>) -> Lane {
    let len = present.keys().next_back().map_or(0, |last| last + 1);
    let cells = (0..len)
        .map(|index| match present.get(&index) {
            Some(&transfer) => Cell::Block { index, transfer },
            None => Cell::Gap { index },
        })
        .collect();
    Lane { side, cells }
}

/// The links today's recipe creates: Donor block N replaces Base block N.
pub fn same_index_links(blocks: &[BlockId]) -> Vec<Link> {
    blocks.iter().map(|b| Link { stack: b.stack.clone(), donor: b.index, base: b.index }).collect()
}

pub const CELL: f32 = 34.0;
pub const GAP: f32 = 4.0;

/// Cells per wrapped line that fit next to the lane labels; never less than one.
pub fn columns(width: f32, label: f32) -> usize {
    (((width - label + GAP) / (CELL + GAP)).floor() as usize).max(1)
}

/// Position ranges of the wrapped lines; the same ranges apply to both lanes.
pub fn lines(width: usize, cols: usize) -> Vec<Range<usize>> {
    (0..width).step_by(cols.max(1)).map(|start| start..(start + cols).min(width)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Both ends on one wrapped line: drawn as a seam between the two columns.
    Same { line: usize, donor_col: usize, base_col: usize },
    /// Ends on different lines: each end is marked with the other end's index instead.
    Split { donor_line: usize, base_line: usize },
}

pub fn place(link: &Link, cols: usize) -> Placement {
    let cols = cols.max(1);
    let (donor_line, base_line) = (link.donor / cols, link.base / cols);
    if donor_line == base_line {
        Placement::Same { line: donor_line, donor_col: link.donor % cols, base_col: link.base % cols }
    } else {
        Placement::Split { donor_line, base_line }
    }
}

#[cfg(test)]
#[path = "tests/schema.rs"]
mod tests;
