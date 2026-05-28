# Testing Target For The Qwen3-TTS Refactor

This repository is moving away from the legacy `tts_core` + `tts_qwen`
architecture. Until code lands, this document defines the intended validation
shape for the new design so later Goal-mode implementation can wire tests to the
correct seams.

## Target Crates

The target workspace is:

- `tts_infer`
- `tts_qwen3_tts`
- `tts_cli`

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
- `SessionSeed<B>` construction
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

## Legacy Test Debt To Remove

When the code migration starts, tests tied only to the deleted architecture
should be removed or rewritten:

- tests centered on `tts_core`
- tests centered on public release/variant routing
- tests that assume `tts_qwen/src/arch`
- tests for fake streaming/chunk policy behavior from the old service loop
