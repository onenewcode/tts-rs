# Testing tts_qwen

## Test Layers

The workspace separates fast Rust-only validation from model-backed synthesis
checks around the engine/session API and the CLI wrapper.

## Fast Validation

```bash
cargo check --workspace
cargo test -p tts_qwen --lib
cargo test -p tts_cli --lib
cargo test -p tts_qwen --test backend_api
cargo test -p tts_qwen --tests --no-run
```

These cover the new engine/session module graph, CLI parsing, and integration
binary compilation without requiring local model weights.

## Model-Backed Integration Tests

The integration tests under `tts_qwen/tests/` still require
`QWEN_TTS_MODEL_DIR`. These tests are useful for
API-level debugging, but the most representative synthesis check is the CLI
smoke run shown below.

Useful runs:

```bash
cargo test -p tts_qwen --test frontend -- --nocapture
cargo test -p tts_qwen --test tokenizer -- --nocapture
cargo test -p tts_qwen --test pipeline -- --nocapture
```

## Preferred CLI End-to-End Smoke

Use the CLI for end-to-end synthesis validation. This exercises backend
selection, engine/session setup, token generation, codec decode, and WAV
writing in the same path that end users call.

```bash
cargo run --release -p tts_cli -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64 \
  --chunk-steps 8
```

This writes `0000.wav` in the current directory.

Why these arguments matter:

- `--language Chinese` and `--speaker Vivian` produce a stable reference path
  for the current local model.
- `--max-new-tokens 64` avoids extremely long generations and trailing silence
  on short prompts.
- `--output-dir .` makes it obvious where the artifact landed when doing manual
  listening checks.

After generation, validate the artifact itself:

```bash
python3 - <<'PY'
import wave
with wave.open("0000.wav", "rb") as wav:
    print("channels=", wav.getnchannels())
    print("rate=", wav.getframerate())
    print("width=", wav.getsampwidth())
    print("frames=", wav.getnframes())
PY
```

Expected shape for the current model path:

- mono channel
- 24000 Hz sample rate
- 16-bit PCM
- non-zero frame count
