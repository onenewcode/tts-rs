# Repository Guidelines

## Project Structure & Module Organization

This workspace has two Rust crates: `tts_qwen/` for inference and `tts_cli/` for the CLI wrapper. Core code in `tts_qwen/src/` is split into `frontend/`, `talker/`, `audio_codec/`, and `shared/`, with public orchestration in `tts_qwen/src/pipeline.rs`. Integration tests live in `tts_qwen/tests/`; focused unit tests sit beside implementation files. See `docs/architecture.md` and `docs/testing_tts_qwen.md` for deeper notes.

## Architecture Overview

```text
text -> frontend -> talker -> codec tokens -> audio_codec -> WAV
                 ^                                    |
                 +--------- pipeline facade ----------+
```

Keep `tts_cli/` thin: parse args, choose backend, call `Qwen3TtsPipeline`, and write output.

## Build, Test, and Development Commands

- `cargo check --workspace` - fast compile check across both crates.
- `cargo test --release --workspace` - default release-mode test suite.
- `cargo test --release -p tts_qwen --test pipeline -- --ignored --nocapture` - real-model E2E smoke test.
- `cargo run --release -p tts_cli --bin tts_cli -- --model-dir Qwen/... --text "你好"` - run local synthesis and write `0000.wav`.
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` - format and lint before review.

Use `QWEN_TTS_MODEL_DIR` or a local `Qwen/*` directory when running model-backed tests.

## Coding Style & Naming Conventions

Follow standard Rust style: 4-space indentation, `snake_case` for functions/modules, `CamelCase` for types, and small focused modules. Keep domain modules depending on `shared`, expose high-level behavior through the pipeline facade, and prefer explicit `thiserror`-based errors.

## Testing Guidelines

Add unit tests next to internal logic and integration tests in `tts_qwen/tests/` for public behavior. Match existing names such as `frontend.rs`, `tokenizer.rs`, and `pipeline.rs`. Keep model-heavy coverage behind ignored tests and document required assets or env vars.

## Commit & Pull Request Guidelines

Recent history uses very short subjects; keep commits imperative and concise, but prefer a bit more context, for example `pipeline: tighten wav validation`. PRs should name changed crate(s), summarize behavior changes, list verification commands, and note model or backend assumptions. Include sample CLI output or generated file paths when touching synthesis flows.

## Configuration Tips

Do not commit local model weights or generated audio artifacts unless explicitly needed. Keep large assets under `Qwen/` locally, and prefer environment-based configuration for machine-specific paths.
