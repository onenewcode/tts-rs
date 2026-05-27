# Testing tts_qwen

## Test Layers

The crate now separates fast Rust-only tests from model-backed integration
checks around the new engine API.

## Fast Validation

```bash
cargo check --workspace
cargo test -p tts_qwen --lib
cargo test -p tts_cli --lib
cargo test -p tts_qwen --tests --no-run
```

These cover the new engine/session module graph, CLI parsing, and integration
binary compilation without requiring local model weights.

## Model-Backed Integration Tests

The integration tests under `tts_qwen/tests/` still require
`QWEN_TTS_MODEL_DIR` or a local `Qwen/*` directory.

Useful runs:

```bash
cargo test -p tts_qwen --test frontend -- --nocapture
cargo test -p tts_qwen --test tokenizer -- --nocapture
cargo test -p tts_qwen --test pipeline -- --nocapture
```

## CLI Smoke

```bash
cargo run --release -p tts_cli -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64 \
  --stream \
  --chunk-steps 8 \
  --profiling
```

This exercises the new `QwenTtsEngine` session loop and writes `0000.wav`.
