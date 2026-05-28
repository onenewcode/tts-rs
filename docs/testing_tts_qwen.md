# Testing Target For The Qwen3-TTS Refactor

This repository now validates the refactored three-crate layout:

- `tts_infer`
- `tts_qwen3_tts`
- `tts_cli`

Use these commands as the default fast verification set:

```bash
cargo test -p tts_infer
cargo test -p tts_qwen3_tts
cargo test -p tts_cli
```

## Service-Layer Validation

Expected fast tests for `tts_infer`:

```bash
cargo test -p tts_infer
```

These should cover:

- `Engine<M>::synthesize()` driving `start_session + step + finish`
- `EngineSession` state guard behavior
- `InferError::Model(...)` vs `InferError::Service(...)`
- `PcmAudio` result invariants

## Model-Crate Validation

Expected fast tests for `tts_qwen3_tts`:

```bash
cargo test -p tts_qwen3_tts --lib
cargo test -p tts_qwen3_tts --tests --no-run
```

These should cover:

- package manifest parsing
- package-path normalization into `Qwen3TtsPackage`
- backend resolution rules
- request validation for `BaseRequest` and `CustomVoiceRequest`
- compiler loading profile config once at engine load
- prompt recipe behavior
- session startup through `start_session + step + finish`
- session finalization into `PcmAudio`

## CLI Validation

Expected fast tests for `tts_cli`:

```bash
cargo test -p tts_cli --lib
```

These should cover:

- package-first input parsing
- profile subcommands
- mapping subcommand args into `QwenRequest`
- mapping run flags into `Qwen3TtsRunOptions`

## Model-Backed Smoke Goal

The preferred end-to-end path is:

- load package through `Qwen3TtsEngine::load(...)`
- synthesize through `Qwen3TtsEngine::synthesize(...)`
- write `PcmAudio` to a WAV file via `tts_cli`

Expected artifact properties:

- mono
- 24000 Hz
- 16-bit PCM
- non-zero frame count

## Model-Backed Optional Check

When local model assets are available, run the ignored real-model smoke test:

```bash
cargo test -p tts_qwen3_tts --test real_model -- --ignored --nocapture
```

This should confirm that package-first loading and in-crate runtime execution
produce mono, 24 kHz, 16-bit PCM output.
