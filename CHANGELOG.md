# Changelog

All notable changes to FrankenForge are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Build report** (`ff_surgery::report`): a block-level account of a plan or build. For every
  block of every stack: output index, source (alias and source block), and status — unchanged,
  shifted, replaced, partially replaced, added or generated; plus removed blocks and tensors,
  tensors outside the stacks with their origin, `config.json` changes, warnings, the experimental
  flag, block counts before/after, and an always-present "model behavior: not evaluated". Text
  output folds runs of identical rows into ranges (`0-14  recipient  unchanged`); JSON has a
  versioned schema (`schema_version: 1`).
- **Weight delta**: for each tensor that replaced a base tensor of the same name,
  `||new - old||_2 / ||old||_2`, and per block over the sum of squares. F32/F16/BF16 only; other
  dtypes, an all-zero base tensor or non-finite values give Unknown, never 0. Streamed in fixed
  4 MiB chunks through `TensorSource::read_range`; a model is never loaded whole.
- CLI: `plan` prints the report (`--json`, `--weights-diff`); `build` prints it with deltas and
  saves `<out>.report.json` and `<out>.report.md` next to the lock file.
- GUI Forge: a **Build report** card with the same block map, statuses, sources, deltas,
  `config.json` changes, warnings and the "behavior not evaluated" note, in en/ru/zh-CN.
- `transplant` step: optional `keep_extra = true` for an intentional partial replacement.
- README opens with the SmolLM2 example (recipe and real report) and describes the Forge and
  Preflight workspaces of the GUI.

### Changed

- `transplant` fails at plan time with `incompatible` when base tensors under `dst` have no donor
  counterpart (they used to stay silently), unless `keep_extra = true`.
- `transplant` into a block index that does not exist yet checks every tensor's dtype and shape
  against the same tensor in the stack's other blocks and fails with `incompatible` on a mismatch.
- `plan` output is the build report instead of the raw tensor listing and provenance counts.
- F16/BF16/F32 decoding moved from the GUI into `ff_core::decode` and is shared by the GUI
  profile and the weight delta.
- ROADMAP/README: "Visual Forge" removed from future plans; it shipped as the GUI's Forge.

## [0.1.0] - 2026-10-25

First public snapshot: an **early prototype**. The engine works and is tested; rebuilt models are
verified on disk, not in behavior.

### Added

- **ff-core** — domain model for checkpoints: `TensorInfo`, `Model`, `DType`, safetensors block
  layout detection (repeated block stacks), a single error type, and the `TensorSource` /
  `ModelFormat` ports. No I/O.
- **ff-io** — safetensors adapter: single file or shard directory; a hardened header reader
  (length/offset/overflow checks on untrusted input) and a writer.
- **ff-surgery** — TOML recipes (`transplant`, `remove`, `rename`, experimental `adapter`) planned
  against metadata only into a `Plan` with per-tensor provenance; atomic builds through
  `<out>.partial`; post-build verification that re-reads and hashes every tensor; a lock file with
  SHA-256 of recipe, sources and output; directory output that carries `config.json` and tokenizer
  files from the base, renumbers stacks and updates the layer count (or fails early with
  `unsupported config`).
- **frankenforge CLI** — `inspect`, `compare` (metadata-only tensor/stack/tokenizer compatibility
  report, text or `--json`), `plan`, `build` (with `--allow-experimental-adapter`).
- **ff-gui** — read-only checkpoint inspector (egui/eframe), the primary user interface:
  - **Learn**: plain-language guided tour for readers without ML background;
  - **Model structure**: components and blocks explained in human terms;
  - **Technical data**: tensor tree, full-name search, `config.json` view;
  - **Tensor Inspector**: role and per-dimension meanings with Confirmed/Inferred/Unknown basis,
    Simple/Expert modes;
  - **Analyze tensor**: streams the single selected tensor in chunks (two passes, bounded memory)
    and reports min/max/mean/std, zeros, NaN/Inf and a 40-bin histogram with progress;
  - localization in English / Русский / 简体中文 (OS auto-detect, saved choice), dark/light theme,
    persisted window settings, bundled Noto Sans SC (OFL 1.1).
- Workspace-wide checks: `cargo fmt`, `cargo clippy -D warnings`, unit + end-to-end tests, an
  ignored byte-exact real-model round-trip test, and CI (fmt/clippy/test/release build on Windows,
  Linux and macOS).
- Open-source packaging: Apache-2.0 license, README, ROADMAP, CONTRIBUTING, SECURITY, code of
  conduct, issue/PR templates, author list, this changelog.

### Known gaps at this version

- No GGUF; no inference; model quality is never evaluated.
- The adapter step writes a weight tensor with no runtime support; builds containing it are
  opt-in and marked NOT validated.
- No `--locked` rebuild; the reader reopens the file per tensor (no mmap).

[Unreleased]: https://github.com/Hemenguelbindi/FrankenForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Hemenguelbindi/FrankenForge/releases/tag/v0.1.0
