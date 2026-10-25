# FrankenForge Roadmap

**Status: Early Prototype (v0.1.0).** This roadmap is the order we intend to work in, not a
release plan. There are no dates and no guarantees: in particular, FrankenForge promises a rebuilt
checkpoint's *integrity and reproducibility*, never its *quality* — a transplanted model may work,
may be broken, and we will tell you which one we actually verified.

## 1. A real SmolLM2 transplant

Everything in v0.1.0 has been exercised on generated fixtures and small local test checkpoints.
The first goal is one honest, complete, public experiment:

- take [SmolLM2](https://huggingface.co/HuggingFaceTB) (a small public model) as base and donor,
- transplant real blocks between two real checkpoints with a committed recipe,
- verify the rebuild byte-for-byte through the lock file,
- publish the recipe, the plan, the hashes and what was learned — including whatever breaks.

Deliverable: a reproducible example (recipe + lock + write-up) in the repository.

## 2. Verification through xInfer

Integrity is not behavior. After SmolLM2 surgery "works on disk", the next step is to load the
rebuilt checkpoint in an actual inference runtime (xInfer) and record what happens: does it run,
does it produce sane output, what does a block swap actually change? This is where the project
earns the right to say "surgery produced a working model" — or documents precisely when it does not.

Deliverable: a documented, repeatable check pipeline from `build` to runtime results. Still no
quality guarantees — only measured, reported outcomes.

## 3. Visual Forge

Today a recipe is TOML. The GUI already explains structure; the next layer is *composing*: pick a
block in model A, drop it into model B, see the compatibility verdict (`compare` already computes
it) and the config consequences live, then export the recipe — and keep the CLI as the execution
surface. The visual editor produces recipes; it does not replace plan/build/verify.

## 4. Read-only GGUF

GGUF is everywhere, so the next format is inspection, not writing: open a GGUF checkpoint in
`inspect` and the GUI, map its metadata and tensors into the same `Model` view. Read-only on
purpose — writing a format we cannot yet verify end-to-end would contradict the project's rule
that "verified" means re-read and hashed.

## 5. Graphs and more complex compositions

Linear recipes (base + steps) give way to a composition graph: multiple donors, cross-architecture
bridges through adapters, chains of block moves with checked port compatibility. This is research
territory; each capability lands only with tests and, where it matters, runtime verification from
step 2.

## Explicit non-goals (for now)

- No dates, no version promises.
- No model-quality guarantees of any kind.
- No training, no quantization, no fine-tuning — FrankenForge recombines checkpoints, it does not
  teach them.
- No writing GGUF before it can be verified round-trip.
