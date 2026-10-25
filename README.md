# FrankenForge

**Early prototype.** FrankenForge is a Rust toolkit for researching and recombining language-model
checkpoints — inspecting what is inside a model, transplanting blocks between models, and exporting
a rebuilt checkpoint that is verified on disk and reproducible from a lock file. No Python is
involved anywhere.

The point of the project is twofold:

- **Make model architecture legible.** Most people who download a `.safetensors` checkpoint cannot
  say what is inside it. FrankenForge opens the file and explains its structure in plain language,
  in the reader's own language, without requiring any ML background.
- **Make experiments reproducible.** Every rebuild is described by a small TOML recipe and ends
  with a lock file containing SHA-256 hashes of the recipe, every source model and the output, so
  the same experiment can be rebuilt and checked bit-for-bit.

Nothing in the current prototype runs a model: the GUI reads metadata only, and surgery verifies
file integrity, not model behavior. See [Limitations](#limitations).

![FrankenForge GUI in English: Technical data view with a tensor selected and the Tensor Inspector analyzing it](examples/app_example.png)

## What works today

### GUI — the main interface for users (`ff-gui`)

A read-only checkpoint inspector built with egui/eframe. It opens a `.safetensors` file or a
directory of shards and reads **metadata only**: whole weights are never loaded into memory and
nothing is ever executed.

- **Three presentations, switched by tabs:**
  - **Learn** — a self-contained guided tour for a reader without ML background: how text becomes
    tokens, tokens become numbers, numbers pass through repeated blocks; with an accordion of
    explanations and diagrams. Facts come from the opened file; missing evidence is stated plainly.
  - **Model structure** — the same checkpoint explained in plain language: components (language /
    audio / vision), repeated blocks, and what each part does (attention, MLP, experts,
    normalization, embeddings, output head…).
  - **Technical data** — the raw view: tensor tree grouped into blocks, full-name search,
    `config.json`, and the **Tensor Inspector**.
- **Tensor Inspector:** selecting a tensor shows its role, the meaning of each dimension
  (`hidden_size`, `vocab_size`, …), and whether each claim is *Confirmed* against `config.json`
  architecture rules, merely *Inferred* from the name, or *Unknown*. Simple/Expert modes.
- **Numeric profile:** the *Analyze tensor* button streams **only the selected tensor** from disk
  in fixed-size chunks (two passes, bounded memory) and computes min/max/mean/std, zeros, NaN and
  infinity counts, and a 40-bin histogram, with progress. Nothing is read until you press the
  button.
- **Localization:** English, Русский, 简体中文 — auto-detected from the OS, switchable in the
  toolbar at any time, saved between runs. The bundled Noto Sans SC font covers all three.
- Dark/light theme; window settings persist.

### CLI — automation, recipes, CI and debugging (`frankenforge`)

The `ff-cli` crate builds on and stays. It is the scripting front for the same engine:

- `inspect` — tensor layout of a file or shard directory;
- `compare` — metadata-only compatibility report of two models: shared/typed-differing/unique
  tensors, block-stack transplant compatibility, tokenizer comparison (`--json` for tools);
- `plan` — apply a recipe to metadata only and print the resulting layout and provenance (dry run);
- `build` — execute the recipe, write to `<out>.partial` and rename only after verification
  (every tensor is re-read and hashed), then write the lock file. Output is either a single
  `.safetensors` file or a directory with `config.json` and tokenizer files copied from the base.

### Surgery operations

| Step | What it does |
|---|---|
| `transplant` | copy a block or sub-block (e.g. an MLP) from a donor model over the same region of the base |
| `remove` | drop tensors under a selector; every layer stack is then renumbered from 0 without gaps |
| `rename` | move tensors under a prefix, e.g. to close layer-index gaps |
| `adapter` | add an explicit `[out_dim, in_dim]` projection bridging two hidden sizes — **experimental**, see below |

`plan` and `build` update `num_hidden_layers` in `config.json` when the block count changes, and
fail with a clear `unsupported config` error **before writing anything** whenever the config pins
per-layer structure (index-dependent keys, nested `text_config`, …) that the tool cannot rewrite
consistently.

## Limitations

Honest state of the prototype:

- **No guarantee that surgery produces a working model.** `build` verifies *file integrity* — bytes,
  dtypes, shapes, hashes — not that a transplanted model runs or produces good output. Model
  behavior is not evaluated in any way.
- **No GGUF support.** Only safetensors (single file or sharded directory) can be read or written.
- **The adapter step is experimental.** It adds a weight tensor only; no runtime executes it and
  there has been no numerical or quality validation. `build` refuses recipes containing an adapter
  unless `--allow-experimental-adapter` is passed, and then marks the output and the lock file
  `experimental` and prints "model NOT validated".
- **No inference.** The GUI and CLI never run a model; the numeric profile describes storage
  statistics, not knowledge or quality.
- `compare` pairs stacks by exact tensor-name prefix only (`model.layers` ≠ `model.language_model.layers`)
  and reads only `tokenizer.json`.
- There is no `--locked` rebuild that re-checks source hashes before building, and the safetensors
  reader reopens the file per tensor (no mmap).

## Quick Start (Windows / PowerShell)

Prerequisites: Git, and a recent stable Rust toolchain with the MSVC backend
(edition 2024 requires Rust 1.85+; install from https://rustup.rs).

```powershell
git clone https://github.com/<owner>/FrankenForge.git
cd FrankenForge

# build the GUI
cargo build --release -p ff-gui

# run it (optionally pass a checkpoint path)
.\target\release\ff-gui.exe
.\target\release\ff-gui.exe C:\models\my-checkpoint
```

The window opens with the guided tour; use *Open file…* or *Open folder…* to load a
`.safetensors` checkpoint or a Hugging Face model directory.

## Inspector Walkthrough

1. **Open a model** — *Open file…* for a single `.safetensors`, *Open folder…* for a checkpoint
   directory (shards, `config.json`, tokenizer). Only headers and metadata are read.
2. **Learn** — the default tab: a short plain-language tour of how a language model works, using
   the structure of *this* file (parameter count, recognized parts, number of repeated blocks).
3. **Model structure** — a map of the file in human terms: each block's parts (attention, MLP,
   experts, normalization…) are labeled with what they do.
4. **Technical data** — the raw tensor tree. Search any tensor by full name; click a tensor to
   select it.
5. **Select a tensor** — the Tensor Inspector panel opens at the right: role, dimension meanings,
   Confirmed/Inferred/Unknown basis, dtype, shape, element count, size on disk.
6. **Analyze** — press *Analyze tensor* to compute the numeric profile (with a progress bar).

**Analyze reads only the selected tensor**, streamed from disk in small chunks; the rest of the
file is never touched, and memory stays bounded however large the tensor is. Statistics are for
floating-point tensors (F16/BF16/F32); the result says how values are *stored* — scale, spread,
zeros, non-finite values — never what the model "knows".

Languages: the interface runs in **English / Русский / 简体中文** (toolbar picker, OS auto-detect,
`--lang en|ru|zh-CN`).

## Advanced: CLI

The CLI is the automation surface: recipes, CI, and debugging. From the workspace root,
`cargo run -- <cmd>` runs the CLI; or build once and call `frankenforge`:

```powershell
cargo build --release -p ff-cli
$ff = '.\target\release\frankenforge.exe'

& $ff inspect C:\models\base                       # file or directory of shards
& $ff compare C:\models\base C:\models\donor       # metadata-only compatibility report
& $ff compare C:\models\base C:\models\donor --json

& $ff plan examples\recipe.toml                    # dry run: resulting layout + provenance
& $ff build examples\recipe.toml -o C:\out\merged.safetensors   # weights only
& $ff build examples\recipe.toml -o C:\out\merged               # directory: weights + config.json + tokenizer

# experimental adapter (no runtime executes it; marks the build NOT validated):
& $ff build adapter-recipe.toml -o C:\out\bridged.safetensors --allow-experimental-adapter
```

A build is finished only when the output has been re-read, every tensor hashed, and
`<out>.safetensors.lock.json` written with SHA-256 of the recipe, every source and the output.
Recipes are TOML with `deny_unknown_fields`; source paths are relative to the recipe file. See
[examples/recipe.toml](examples/recipe.toml) for every step type.

There is one ignored end-to-end test that checks a byte-exact no-op rebuild against a real
checkpoint you provide:

```powershell
$env:FF_REAL_MODEL = 'C:\models\qwen3-moe-tiny-random'
cargo test -p ff-cli --test real_roundtrip -- --ignored
```

## Repository layout

Dependencies point inward only: `ff-cli`/`ff-gui` → adapters → `ff-core`.

```
FrankenForge/
├── crates/
│   ├── ff-core/      # domain: TensorInfo, Model, DType, block layout, errors, ports. No I/O.
│   ├── ff-io/        # format adapters implementing the ports. Today: safetensors.
│   ├── ff-surgery/   # use cases: recipe → Plan (metadata + provenance) → ops → build → verify → lock.
│   ├── ff-cli/       # `frankenforge` — composition root for automation.
│   └── ff-gui/       # `ff-gui` — composition root for the main user interface (egui/eframe).
├── examples/         # recipe.toml (every step type), GUI screenshot
└── Cargo.lock        # tracked: the app is a shipped binary
```

A new format (GGUF, …) is a new module in `ff-io` implementing `ModelFormat`; a new operation is a
`Step` variant plus a file in `ff-surgery/src/ops/`. `ff-surgery` knows only the ports — planning
touches metadata, weights are streamed one tensor at a time.

## Roadmap

Short version, in order (details in [ROADMAP.md](ROADMAP.md)):

1. A **real SmolLM2 transplant** — end-to-end surgery on a public model, recipe and lock file included.
2. **Verification through xInfer** — actually run a rebuilt checkpoint in an inference runtime.
3. **Visual Forge** — build recipes as a visual composition in the GUI, not only in TOML.
4. **Read-only GGUF** — inspect GGUF checkpoints without claiming to write them.
5. **Graphs and more complex compositions** — multi-input surgery beyond pairwise block swapping.

No dates and no promises: a rebuilt model's *quality* is not something this project guarantees —
only its integrity and reproducibility.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: Rust only (no Python anywhere), tests for
every new branch, English code comments, GUI strings as Fluent keys in all three locales, and a
clean-room rule — implement from published format specifications, never by copying from competing
tools. Run `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace` before opening a PR.

Ideas and which step comes next live in [ROADMAP.md](ROADMAP.md); pick an open issue or open a new
one first for anything beyond a small fix.

## Security

FrankenForge parses untrusted checkpoint files. How we treat that, what is in scope, and how to
report a vulnerability privately: [SECURITY.md](SECURITY.md).

## License

Code is licensed under the [Apache License 2.0](LICENSE).

The bundled UI font `crates/ff-gui/assets/fonts/NotoSansSC-Regular.otf` is licensed under the
SIL Open Font License 1.1 — see [`OFL.txt`](crates/ff-gui/assets/fonts/OFL.txt) next to it. That
license applies to the font only and is preserved on redistribution.

## AI-assisted development

This project was developed with AI assistance (an LLM coding agent wrote most of the code under
human direction, review and testing). The maintainers treat AI output as untrusted until it passes
the checks above; every claim in this README describes code that is in this repository and tested.
Contributions that use AI tools are welcome under the same bar: you reviewed every line you submit.
