# Contributing to FrankenForge

Thanks for your interest. FrankenForge is an early prototype; the bar below keeps it honest and
maintainable.

## Ground rules

- **Rust only.** No Python anywhere — not in tooling, tests, fixtures or helper scripts. Test
  models are generated in Rust (see `crates/ff-cli/tests/build_e2e.rs`).
- **English code comments.** All comments, doc comments and CLI help are in English. Keep comments
  to the *why* the code cannot show; known limits get `// shortcut: <limit>, <when to upgrade>`.
- **Every new branch of logic gets a test.** Unit tests for ops and parsers live next to the code
  by convention: for `src/<dir>/<name>.rs` they are in `src/<dir>/tests/<name>.rs` (declared at the
  bottom of the source file); integration tests in `crates/<crate>/tests/`.
- **GUI strings are never hard-coded.** Every user-visible string is a Fluent message in
  `crates/ff-gui/locales/{en,ru,zh-CN}.ftl`; add the key to all three files (tests fail on a
  missing or extra key, and on any character the bundled font lacks).
- **Architecture is inward-only:** `ff-cli`/`ff-gui` → `ff-surgery`/`ff-io` → `ff-core`.
  `ff-surgery` talks to formats only through the ports in `ff-core/src/ports.rs`. Planning touches
  metadata only; weights are streamed one tensor at a time. `ff-gui` displays; it parses nothing on
  its own.
- **Don't silently lower the bar.** Existing checks, validation and error handling must be kept or
  improved. A PR that disables a test or loosens a check needs an explanation and will likely be
  refused.

## Clean room: no copying from competitors

FrankenForge must remain free of other tools' code and data:

- Implement file formats from **published specifications** (e.g. the safetensors and GGUF specs)
  and your own reading of the code in *this* repository.
- **Do not copy source code**, decompiled output, or substantial structures from other model
  editors, converters or merge tools — even "lightly adapted".
- Do not commit decompiled third-party binaries, game/model files, or data scraped from other
  projects' repositories.
- If you port an algorithm from a paper or spec, cite it in a comment; if you had to read another
  tool's source to understand a behavior, say so in the PR so it can be reviewed as a re-implementation.

## Before you open a PR

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p ff-gui   # when GUI code changed
```

All four must pass; CI runs the same set on Windows, Linux and macOS (Linux and macOS legs are
new — the maintainers develop on Windows, so first CI runs may need dependency-list fixes).

## Workflow

1. Pick or open an issue first for anything beyond a small fix (the ROADMAP lists the intended
   order of work — comment there before starting a big feature).
2. Branch from `main`, keep PRs focused; one logical change per PR.
3. Describe *what* and *why* in the PR body; link issues; update `CHANGELOG.md` under
   `Unreleased` for user-visible changes.
4. Do not commit models, checkpoints, build outputs, `.partial` files, personal paths or secrets.
   `.gitignore` covers the common cases; check `git status` before committing.

## Style notes

- One error type: `ff_core::Error` / `ff_core::Result`; attach paths to I/O errors with
  `.map_err(Error::io(path))`. No `unwrap()` outside tests.
- Selector semantics (`ff_core::selects`): a string ending in `.` is a block prefix, anything else
  is one exact tensor name. Keep every op on this rule.
- Anything read from a model file is untrusted: check lengths, offsets and overflow before
  allocating.
- Writes go to `<out>.partial` and are renamed on success; a build is not done until `verify`
  re-reads the output and the lock file is written.
- File size: aim for ≤300 lines, hard limit 500 — split by responsibility before crossing it.

## Code of conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
