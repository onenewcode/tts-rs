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


## Base / CustomVoice Validation Status

Current repo-level validation status:

- `CustomVoice`
  - local smoke baseline exists for `text + language + speaker`
  - `instruct` support is planned and must be re-smoked after implementation
- `Base`
  - voice-clone support is planned
  - repo-level Base smoke has not been completed yet
  - until local Base smoke passes, Base must be reported as `unverified` or `experimental`

## Additional Smoke Acceptance

### CustomVoice

After `instruct` lands, the implementation is only accepted when:

- CLI path can synthesize with `--speaker`
- CLI path can synthesize with `--speaker` + `--instruct`
- output WAV is mono, 24 kHz, 16-bit PCM, and non-empty

### Base

Base support is only accepted as repo-verified when all of the following are true:

- local Base weights load through `Qwen3TtsEngine::load(...)`
- local reference WAV can be consumed through the new Base clone path
- `ref_audio + ref_text` synthesis produces a non-empty WAV
- `--x-vector-only` synthesis also runs successfully at least once
- output WAV is mono, 24 kHz, 16-bit PCM, and non-empty

Until that happens, Base should be treated as implemented-but-unverified rather than complete.


## Verification Checklist

Use this checklist before reporting completion.

### CustomVoice checklist

- request/API path supports `instruct`
- CLI path supports `--instruct`
- existing `speaker` path still works
- ignored smoke or manual smoke produces a non-empty WAV
- output WAV is mono / 24 kHz / 16-bit PCM

### Base checklist

- request/API path supports clone input or prepared prompt
- CLI path supports `--ref-audio`
- CLI path supports `--ref-text`
- CLI path supports `--x-vector-only`
- local WAV preprocessing works
- create prompt helper works
- `ref_audio + ref_text` synthesis succeeds
- `--x-vector-only` synthesis succeeds
- output WAV is mono / 24 kHz / 16-bit PCM
- Base status is not reported as verified unless both smoke paths pass

## Verification Report Minimum

Any completion report for this work should include at least:

- exact commands run
- pass/fail result for each command
- whether Base is `unverified`, `implemented-but-unverified`, or `repo-verified`
- generated artifact paths for successful smoke runs
