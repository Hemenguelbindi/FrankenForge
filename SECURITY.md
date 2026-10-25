# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (current) | ✅ |
| < 0.1 | ❌ (pre-release) |

FrankenForge is an early prototype. Until 1.0 we fix security issues on `main` and note them in
`CHANGELOG.md`; backporting to older tags is best-effort.

## What FrankenForge actually does — and its attack surface

- It **parses untrusted files**: `.safetensors` headers and shard directories from the internet are
  hostile input by default. Every length, offset and size read from a file is bounds-checked before
  anything is allocated — a crafted checkpoint must fail, not crash or corrupt memory.
- It **writes only where you point it**: builds go to an explicit `-o` path via `<out>.partial` +
  rename; existing output directories are never overwritten.
- It **never executes model code and never networks**: the GUI reads metadata only; `Analyze`
  streams the single selected tensor's bytes and computes statistics. There is no telemetry, no
  update checks, no remote code path in the tool itself.
- It **does not sandbox model content**: a rebuilt checkpoint carries the base and donor weights;
  running it later in some runtime is your risk and your responsibility.

## Known limitations (not vulnerabilities)

- No memory-mapped reads; large-file handling is chunked but the reader reopens files per tensor.
- Malformed `config.json` is shown as-is in the GUI (it is parsed as JSON for display and layer
  count only, with an explicit error state for invalid files).

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

1. Report privately via a [GitHub security advisory](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/creating-a-repository-security-advisory)
   on this repository ("Report a vulnerability").
2. If advisories are not yet enabled, contact the maintainer listed in `AUTHORS.md` through any
   private channel available on your GitHub profile.

We aim to acknowledge within 5 business days during the prototype phase, and will coordinate a fix
and disclosure timeline with you. Please include: FrankenForge version (`frankenforge --version` /
`ff-gui` title bar), OS, a description, and — if possible — the offending checkpoint file or a
minimal reproduction.
