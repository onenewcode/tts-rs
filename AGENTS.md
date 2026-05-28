# Repository Guidelines

## Project Structure & Module Organization

This workspace has two Rust crates: `tts_qwen/` for inference and `tts_cli/` for the CLI wrapper. Core inference code in `tts_qwen/src/` is organized around `arch/`, `profile/`, `runtime/`, `io/`, and `profiling/`. Public orchestration now lives in `tts_qwen/src/lib.rs` and `tts_qwen/src/backend.rs`; `tts_cli/` should remain a thin adapter over that API surface. Integration tests live in `tts_qwen/tests/`; focused unit tests sit beside implementation files. See `docs/architecture.md` and `docs/testing_tts_qwen.md` for deeper notes.

## Architecture Overview

```text
text -> frontend runner -> talker/model -> codec tokens -> audio codec -> WAV
                        ^                                              |
                        +------ runtime/backend family facade --------+
```

Keep `tts_cli/` thin: parse args, choose backend, call the `tts_qwen` facade, and write output.

## Build, Test, and Development Commands

Use the workspace-standard Cargo commands for build, formatting, lint, and test workflows. Keep detailed test and smoke-run procedures in `docs/testing_tts_qwen.md`, including required model-backed setup such as `QWEN_TTS_MODEL_DIR`.

## Coding Style & Naming Conventions

Follow standard Rust style: 4-space indentation, `snake_case` for functions/modules, `CamelCase` for types, and small focused modules. Keep domain modules depending on `shared`, expose high-level behavior through the pipeline facade, and prefer explicit `thiserror`-based errors.

## Testing Guidelines

Add unit tests next to internal logic and integration tests in `tts_qwen/tests/` for public behavior. Match existing names such as `frontend.rs`, `tokenizer.rs`, and `pipeline.rs`. Keep model-heavy coverage behind ignored tests and document required assets or env vars in `docs/testing_tts_qwen.md`.

## Commit & Pull Request Guidelines

Recent history uses very short subjects; keep commits imperative and concise, but prefer a bit more context, for example `pipeline: tighten wav validation`. PRs should name changed crate(s), summarize behavior changes, list verification commands, and note model or backend assumptions. Include sample CLI output or generated file paths when touching synthesis flows.

## Configuration Tips

Do not commit local model weights or generated audio artifacts unless explicitly needed. Keep large assets under `Qwen/` locally, and prefer environment-based configuration for machine-specific paths.
