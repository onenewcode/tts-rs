# Repository Guidelines

## Project Structure & Module Organization

This workspace now has three Rust crates: `tts_infer/` for the session-layer
inference contract, `tts_qwen3_tts/` for the concrete Qwen3-TTS model runtime,
and `tts_cli/` for the CLI wrapper. In `tts_qwen3_tts/src/`, request and
package-facing code lives under `request/`, `compiler/`, `package/`,
`profiling/`, `runtime/`, and `model/`. `tts_cli/` should remain a thin adapter
over that API surface. See `docs/architecture.md`, `docs/testing_tts_qwen.md`,
and `docs/refactor/` for the current target-state notes.

## Architecture Overview

```text
text -> request compiler -> talker/model -> codec tokens -> audio codec -> WAV
                         ^                                              |
                         +------ tts_infer session/runtime facade ------+
```

Keep `tts_cli/` thin: parse args, choose backend, call the `tts_qwen3_tts`
facade, and write output.

## Build, Test, and Development Commands

Use the workspace-standard Cargo commands for build, formatting, lint, and test
workflows. Keep detailed test and smoke-run procedures in
`docs/testing_tts_qwen.md`, including the model-backed ignored smoke test for
local `Qwen/` assets.

## Coding Style & Naming Conventions

Follow standard Rust style: 4-space indentation, `snake_case` for
functions/modules, `CamelCase` for types, and small focused modules. Keep
`tts_infer` limited to stable service-layer concerns, expose high-level model
behavior through the `tts_qwen3_tts` facade, and prefer explicit
`thiserror`-based errors.

## Testing Guidelines

Add unit tests next to internal logic and integration tests in the owning crate
for public behavior. Keep fast checks under `tts_infer/tests/`,
`tts_qwen3_tts/tests/`, and `tts_cli/tests/`. Keep model-heavy coverage behind
ignored tests and document required assets or env vars in
`docs/testing_tts_qwen.md`.

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
