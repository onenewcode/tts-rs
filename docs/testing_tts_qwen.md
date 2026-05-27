# Testing tts_qwen

## Test Layers

The workspace keeps a compact release-mode test surface for the current V9
path: Rust unit tests plus Python tokenizer/prefill oracles.

## Fast Tests

Run the default suite:

```bash
cargo test --release -p tts_qwen
cargo test --release -p tts_cli
```

Useful focused runs:

```bash
cargo test --release -p tts_qwen talker::
cargo test --release -p tts_qwen audio_codec::
```

The tokenizer/prefill integration tests require the local Qwen model directory
and dynamically invoke the retained Python oracle scripts.

## CLI Smoke

Run the command-line path through the standalone CLI crate:

```bash
cargo run --release -p tts_cli --bin tts_cli -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64
```

This writes `0000.wav` in the requested output directory.

## V9 Alignment Tests

The retained Python alignment checks are:

```bash
cargo test --release -p tts_qwen --test alignment_prefill -- --nocapture
cargo test --release -p tts_qwen --test alignment_tokenizer -- --nocapture
```
