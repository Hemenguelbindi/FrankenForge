# FrankenForge

Rust tool for inspecting and surgically recombining LLM checkpoints (transplant blocks, explicit
adapters, verified export, reproducible lock file). **No Python anywhere** — not in the tool, tests,
fixtures or helper scripts. Test models are generated in Rust (see `crates/ff-cli/tests/build_e2e.rs`).

## Commands

```sh
cargo test --workspace                     # unit tests + end-to-end build test
cargo clippy --workspace --all-targets     # must stay warning-free
cargo run -- inspect <model>               # file or directory of .safetensors shards
cargo run -- compare <a> <b> [--json]      # совместимость двух моделей, только метаданные
cargo run -- plan examples/recipe.toml     # dry run, metadata only
cargo run -- build <recipe> -o out.safetensors
cargo run -p ff-gui -- [<model>]           # GUI inspector; release: cargo build --release -p ff-gui
```

Toolchain: Rust 1.97, edition 2024 (let-chains are used). Shared deps are declared once in the root
`[workspace.dependencies]`; crates reference them with `.workspace = true`.

## Architecture (dependencies point inward only)

```
ff-cli  →  ff-surgery  →  ff-core  ←  ff-io
```

| Crate | Layer | May depend on |
|---|---|---|
| `ff-core` | domain: `TensorInfo`, `Model`, `DType`, `layout`, `Error`, ports `TensorSource` / `ModelFormat` | `serde`, `thiserror` only. **No I/O.** |
| `ff-io` | format adapters implementing the ports (safetensors today) | `ff-core` |
| `ff-surgery` | use cases: recipe → `Plan` → `ops` → build → verify → lock | `ff-core` only — **never `ff-io`** |
| `ff-cli` | composition root: picks the format, calls use cases, prints | everything |
| `ff-gui` | composition root: egui/eframe Inspector (read-only, metadata only, no inference) and Forge (builds through ff-surgery) | `ff-core`, `ff-io`, `ff-surgery` |

Rules:
- `ff-surgery` works through `&dyn ModelFormat` / `&dyn TensorSource`. Need a format feature there?
  Extend the port in `ff-core/src/ports.rs`, implement it in `ff-io`.
- Planning touches metadata only. Weights are read only in `build::build`, one tensor at a time via
  the `fetch` callback — never load a whole model into memory.
- `ff-cli` holds no logic beyond argument parsing and printing.
- `ff-gui` Inspector only displays: opening goes through `ModelFormat::open` (headers only), structure
  through `Model::layout`. Forge calls ff-surgery use cases (`compare::run`, `build::plan`,
  `build::build`) and never duplicates them. No parsing, surgery or compare logic of its own; view
  state that can be tested without egui lives in `ff-gui/src/inspected.rs` and `ff-gui/src/forge/state.rs`.
- `ff-gui` text is never hard-coded: every user-visible string is a Fluent message in
  `crates/ff-gui/locales/{en,ru,zh-CN}.ftl`, read via `I18n::t`/`ta`/`count`. Add a new key to all
  three files (tests fail on a missing or extra key, and on any character the bundled font lacks).
  Tensor names, config keys, paths and core error text are shown as-is, never translated; core errors
  get a localized explanation plus the original text under "Technical details".
- The bundled font `crates/ff-gui/assets/fonts/NotoSansSC-Regular.otf` is SIL OFL 1.1 (`OFL.txt` next to it).

## Where things go

- New checkpoint format (GGUF, ...): new module under `crates/ff-io/src/<format>/`, implement
  `ModelFormat`, re-export from `ff-io/src/lib.rs`.
- New surgery operation: a `Step` variant in `ff-surgery/src/recipe.rs` + one file in
  `ff-surgery/src/ops/` + a match arm in `ops/mod.rs`. Validate compatibility in the op (at plan
  time), not during writing.
- New output tensor kind not copied from a source: a `Payload` variant in `plan.rs`, materialized
  in `build::fetch`.

## Conventions

- **File size: aim for ≤300 lines, hard limit 500.** Split by responsibility into a submodule
  directory (`foo/mod.rs` + parts) before crossing it.
- One error type: `ff_core::Error` / `ff_core::Result`. Attach paths to I/O errors with
  `.map_err(Error::io(path))`. No `unwrap()` outside tests; `expect` only for invariants that cannot
  fail (serializing plain structs), with a message saying why.
- Selector semantics (`ff_core::selects`): a string ending in `.` is a block prefix, anything else is
  one exact tensor name. Keep every op on this rule so `layers.1.` never matches `layers.10.`.
- Recipes are TOML with `deny_unknown_fields`; a typo must fail, not be ignored. Paths in a recipe
  are relative to the recipe file (`Recipe::base_dir`).
- Anything read from a model file is untrusted: check header length, offsets, size overflow
  (see `ff-io/src/safetensors/header.rs`) before allocating.
- Writes go to `<out>.partial` and are renamed on success; never leave a truncated output.
- A build is not done until `verify` re-reads the output and the lock file is written.
- Code comments (including doc comments and clap help) are in English; don't comment the obvious.
- Doc comments explain *why*, briefly. Known limits get `// shortcut: <limit>, <when to upgrade>`.
- Every new op or parser branch gets a unit test. **Тесты не пишутся в исходном файле**: для модуля
  `<dir>/<name>.rs` они лежат в `<dir>/tests/<name>.rs` (для `<dir>/mod.rs` — `<dir>/tests/<dir>.rs`),
  а в конце исходника только объявление:
  ```rust
  #[cfg(test)]
  #[path = "tests/<name>.rs"]
  mod tests;
  ```
  Так тесты остаются дочерним модулем (`use super::*`, доступ к приватному). Интеграционные тесты
  крейта — в `crates/<crate>/tests/`. In-memory источники: `ff_surgery::testutil::{Fake, model}`.

## Known gaps

GGUF; `compare` pairs stacks by exact name only and reads only `tokenizer.json`; `--locked` rebuild that checks source hashes;
`config.json`: only the layer count is updated (no hidden size, no per-layer arrays, no nested
`text_config`); adapters add a weight tensor
only — nothing runs inference yet, so `build` refuses them without `--allow-experimental-adapter`
and marks the lock file `experimental` (keep it that way until a runtime and validation exist); the reader reopens the file per tensor (no mmap).
