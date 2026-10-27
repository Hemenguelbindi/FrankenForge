# FrankenForge Roadmap

**Status: Early Prototype (v0.1.1).** This roadmap is the order we intend to work in, not a
release plan. There are no dates and no guarantees: in particular, FrankenForge promises a rebuilt
checkpoint's *integrity and reproducibility*, never its *quality* — a transplanted model may work,
may be broken, and we will tell you which one we actually verified.

## Done

### A real SmolLM2 transplant (v0.1.1)

Block 15 of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) replaced with block
15 of SmolLM2-135M-Instruct: the committed recipe
[examples/smollm2-transplant.toml](examples/smollm2-transplant.toml), a byte-for-byte check through
the lock file, a real-model test (`crates/ff-cli/tests/real_smollm2_transplant.rs`) and the
block-level build report shown in the README. What was learned: the Instruct fine-tune moved every
block by a similar 7–9.5% (most in blocks 21–28), so one transplanted block carries only a small
part of what the fine-tune changed. Whether the result still answers well is not measured yet —
that is the next step.

### Visual composition: Forge (v0.1.1)

Formerly planned as "Visual Forge", now the GUI's **Forge** workspace: pick a Base and a Donor,
see per-block structural compatibility and where their weights differ, take Donor blocks into the
same positions, read the plan and the build report, and build through the same
plan/build/verify/lock path as the CLI (the recipe is written next to the result). Not there yet:
transfers between different positions, which need an explicit, checked position mapping.

### Seeing what is inside a model (v0.1.1)

The model as a graph of its parts sized by parameters, with the largest part, entry, exit and
most linked part found automatically; and the weights read in words without running anything:
words a word table places next to a word, words the strongest neurons of a block push toward.

## 1. Running models and comparing answers

Integrity is not behavior. The next step is to run small rebuilt checkpoints (SmolLM2 first, in
Rust) next to their base and donor on the same prompts and record what happens: does it run, does
it answer sensibly, what does a block swap actually change? This is where the project earns the
right to say "surgery produced a working model" — or documents precisely when it does not.

Deliverable: a repeatable check from `build` to answers side by side. Still no quality
guarantees — only measured, reported outcomes.

## 2. Read-only GGUF

GGUF is everywhere, so the next format is inspection, not writing: open a GGUF checkpoint in
`inspect` and the GUI, map its metadata and tensors into the same `Model` view. Read-only on
purpose — writing a format we cannot yet verify end-to-end would contradict the project's rule
that "verified" means re-read and hashed.

## 3. Graphs and more complex compositions

Linear recipes (base + steps) give way to a composition graph: multiple donors, cross-architecture
bridges through adapters, chains of block moves with checked port compatibility. This is research
territory; each capability lands only with tests and, where it matters, runtime verification from
step 1.

## Explicit non-goals (for now)

- No dates, no version promises.
- No model-quality guarantees of any kind.
- No training, no quantization, no fine-tuning — FrankenForge recombines checkpoints, it does not
  teach them.
- No writing GGUF before it can be verified round-trip.
