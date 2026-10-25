# Contributing to FrankenForge

Thanks for your interest. FrankenForge is an early prototype; the bar below keeps it honest and
maintainable.

## Contributions and licensing

- You keep the copyright on your contribution. By submitting it, you confirm that you have the
  right to publish it and you agree that it is distributed under the project's
  [Apache License 2.0](LICENSE).
- You are responsible for understanding, reviewing and testing what you submit. Review and merge
  by the maintainers do not transfer authorship of your contribution to them.
- AI-assisted contributions are welcome under the same bar: you must understand and have tested
  every line you submit, and disclose significant use of AI tools in the PR description.
- Unreviewed generated code is not accepted.

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

## Clean room: no copying from competing tools

This rule is about **competing model-editing tools**, not about third-party code in general:

- Declared third-party dependencies and bundled assets (e.g. the OFL-licensed UI font) are fine —
  use them and respect their licenses.
- **Do not copy code or data** from other model editors, converters or merge tools — including
  decompiled output and substantial structures, even "lightly adapted".
- Do not read a competing tool's source in order to reproduce its behavior. If you came into such
  contact accidentally, disclose it to the maintainers **before** continuing the related
  implementation.
- Do not commit decompiled third-party binaries, checkpoints, or data scraped from other projects'
  repositories.
- Implement file formats from **published specifications** (e.g. the safetensors and GGUF specs).
- Algorithms from public papers and specifications may be implemented — cite the source in a
  comment.

## Before you open a PR

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p ff-gui   # when GUI code changed
```

All four must pass. CI runs the same set across the OS matrix defined in
[.github/workflows/ci.yml](.github/workflows/ci.yml); see that file for the current platforms and
commands.

## Workflow

1. Check ROADMAP.md before starting a large feature, then open or join an issue to discuss the
   approach.
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
