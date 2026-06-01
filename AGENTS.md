# Repository Guidelines

## Project Structure & Module Organization

This workspace has five Rust crates: `tts_infer/` (package `tts_core`) for the
framework core, `tts_error/` for shared diagnostics, `tts_qwen3_tts/` for the
concrete Qwen3-TTS driver, `tts_app/` for application-service orchestration,
and `tts_cli/` for the CLI shell.

In `tts_qwen3_tts/src/`, the current primary layers are `surface/`, `loading/`,
`capabilities/`, and `execution/`. Model-private internals still live under
`model/`, and some backend concerns are still handled by existing modules while
the internal split settles. `tts_cli/` should remain a thin adapter over
`tts_app`.

See `docs/architecture.md` for the implementation split and `docs/TEST.md` for
verification flows.

## Architecture Overview

```text
text -> request preparation -> talker/model -> codec tokens -> audio codec -> WAV
                           ^                                          |
                           +------ tts_core loaded-model lifecycle ---+
```

Keep `tts_cli/` thin: parse args, call `tts_app`, and report output.

## Build, Test, and Development Commands

Use the workspace-standard Cargo commands for build, formatting, lint, and test
workflows. Keep detailed verification and smoke-run procedures in
`docs/TEST.md`, including the model-backed CLI release-mode smoke path for
local `Qwen/` assets.

## Coding Style & Naming Conventions

Follow standard Rust style: 4-space indentation, `snake_case` for
functions/modules, `CamelCase` for types, and small focused modules. Keep
`tts_core` limited to stable framework concerns, expose high-level model
behavior through `tts_app` and the `tts_qwen3_tts` surface, and prefer explicit
`thiserror`-based errors with shared diagnostics in `tts_error`.

## Testing Guidelines

Add unit tests next to internal logic and integration tests in the owning crate
for public behavior. Keep fast checks under `tts_infer/tests/`, `tts_app/tests/`,
`tts_qwen3_tts/tests/`, and `tts_cli/tests/`. Keep model-heavy coverage behind
ignored tests or documented smoke procedures, and record any required local
assets or env vars in `docs/TEST.md`.

## Commit & Pull Request Guidelines

Recent history uses very short subjects; keep commits imperative and concise,
but prefer a bit more context, for example `compiler: re-home session seed
logic`. PRs should name changed crate(s), summarize behavior changes, list
verification commands, and note model or backend assumptions. Include sample CLI
output or generated file paths when touching synthesis flows.

## Configuration Tips

Do not commit local model weights or generated audio artifacts unless explicitly
needed. Keep large assets under `Qwen/` locally, and prefer environment-based
configuration for machine-specific paths.
