# Repository Guidelines

## Project Structure & Module Organization

This workspace now has five Rust crates: `tts_infer/` (package `tts_core`) for
the framework core, `tts_error/` for shared diagnostics, `tts_qwen3_tts/` for
the concrete Qwen3-TTS driver, `tts_app/` for application-service orchestration,
and `tts_cli/` for the CLI shell. In `tts_qwen3_tts/src/`, the target internal
layers are `surface/`, `loading/`, `capabilities/`, `execution/`, and
`backend/`; legacy support modules may still exist behind those boundaries while
the refactor settles. `tts_cli/` should remain a thin adapter over `tts_app`.
See `docs/architecture.md`, `docs/testing_tts_qwen.md`, and `docs/refactor/`
for the current target-state notes.

## Architecture Overview

```text
text -> request compiler -> talker/model -> codec tokens -> audio codec -> WAV
                         ^                                              |
                         +-------- tts_core loaded-model facade -------+
```

Keep `tts_cli/` thin: parse args, call `tts_app`, and report output.

## Build, Test, and Development Commands

Use the workspace-standard Cargo commands for build, formatting, lint, and test
workflows. Keep detailed test and smoke-run procedures in
`docs/testing_tts_qwen.md`, including the model-backed ignored smoke test for
local `Qwen/` assets.

## Coding Style & Naming Conventions

Follow standard Rust style: 4-space indentation, `snake_case` for
functions/modules, `CamelCase` for types, and small focused modules. Keep
`tts_core` limited to stable framework concerns, expose high-level model
behavior through `tts_app` and the `tts_qwen3_tts` surface, and prefer explicit
`thiserror`-based errors with shared diagnostics in `tts_error`.

## Testing Guidelines

Add unit tests next to internal logic and integration tests in the owning crate
for public behavior. Keep fast checks under `tts_infer/tests/`,
`tts_app/tests/`, `tts_qwen3_tts/tests/`, and `tts_cli/tests/`. Keep
model-heavy coverage behind ignored tests and document required assets or env
vars in `docs/testing_tts_qwen.md`.

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
