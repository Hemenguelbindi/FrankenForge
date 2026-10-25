# What & why

<!-- One-paragraph summary of the change and the problem it solves. Link issues: "Fixes #123". -->

# How it was verified

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] `cargo build --release -p ff-gui` passes (when GUI code changed)
- [ ] New/changed logic has tests (unit tests in `<dir>/tests/<name>.rs`, or integration tests)
- [ ] GUI strings: added to **all three** locales (`en`, `ru`, `zh-CN`) — no hard-coded text

# Hygiene

- [ ] No models, checkpoints, `.partial` files, build outputs or secrets committed
- [ ] No personal paths (usernames, private drive layouts) in code, docs or test fixtures
- [ ] Clean-room: implemented from published specs and this repo only, no code copied from
      competing tools

# Notes for the reviewer

<!-- Anything surprising, anything you deliberately left as a `shortcut:`. -->
