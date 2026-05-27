# Testing tts_qwen

## Test Layers

The workspace keeps a compact release-mode Rust-only test surface for the
current V9 path: unit tests, tokenizer/prefill behavior tests, and an ignored
real-model pipeline smoke.

## Fast Tests

Run the default suite:

```bash
cargo test --release --workspace
```

Useful focused runs:

```bash
cargo test --release -p tts_qwen talker::
cargo test --release -p tts_qwen audio_codec::
cargo test --release -p tts_qwen --test tokenizer
cargo test --release -p tts_qwen --test prefill
```

The model-dependent Rust tests require `QWEN_TTS_MODEL_DIR` or a local
`Qwen/*` model directory under the workspace root.

## CLI Smoke

Run the command-line path through the standalone CLI crate:

```bash
cargo run --release -p tts_cli --bin tts_cli -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64 \
  --log-level info
```

This writes `0000.wav` in the requested output directory.

## Real E2E Smoke

The full Rust pipeline smoke loads real weights, generates audio, and validates
WAV metadata plus non-zero PCM data:

```bash
cargo test --release -p tts_qwen --test pipeline -- --ignored --nocapture
```
