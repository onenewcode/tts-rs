# Testing Strategy For TTS-RS

## Summary

Testing must follow the target architecture, not only the current file layout.

The repository now exposes five crates, and the test strategy must reinforce
the implemented framework boundaries:

- framework-core behavior stays fast and deterministic
- Qwen3 driver behavior is split between public-surface tests and deeper driver
  tests
- model-backed smoke paths stay ignored and explicitly documented
- CLI tests only verify shell behavior and argument parsing

## Test Layers

### 1. Core fast tests

Current location:

- `tts_infer/tests/` (package `tts_core`)

Must verify:

- engine/session lifecycle rules
- step/finish contract enforcement
- audio serialization primitives
- instance lifecycle state transitions once framework core is introduced

Should stay:

- fast
- deterministic
- asset-free

### 2. Driver public-surface tests

Current location:

- `tts_qwen3_tts/tests/public_surface.rs`
- `tts_qwen3_tts/tests/compiler_load.rs`

Must verify:

- public request defaults
- package source normalization
- backend parsing and selection behavior
- public load/config defaults
- Qwen3 public capability projection once added

Should avoid:

- real model weights
- large tensor execution

### 3. Driver internal behavior tests

Current location:

- unit tests under `tts_qwen3_tts/src/`
- graph/spec tests under the model tree

Must verify:

- request pre-processing invariants
- compilation invariants
- execution-stage contracts
- capability aggregation rules
- backend availability logic

### 4. CLI end-to-end smoke verification

Current location:

- CLI runtime against local `Qwen/` assets

Must verify:

- custom-voice synthesis from the CLI shell
- optional base-model synthesis paths when the shell needs them

Rules:

- do not require these runs in default CI-style fast checks
- document local asset assumptions explicitly

### 5. CLI shell tests

Current location:

- `tts_cli/tests/cli_parse.rs`

Must verify:

- clap parsing
- command shape
- shell argument compatibility

Should not verify:

- model internals
- request semantic correctness after framework/application services absorb
  request assembly

## Required Verification Commands

Fast checks:

```bash
cargo test -p tts_core
cargo test -p tts_qwen3_tts --test public_surface
cargo test -p tts_qwen3_tts --test compiler_load
cargo test -p tts_cli --test cli_parse
```

Current repo note:

- `cargo test -p tts_core` is the framework-core fast check in the current repository state

CLI end-to-end smoke path:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --backend flex \
  --output ./out/custom-voice-flex-smoke.wav
```

CUDA CLI verification:

```bash
cargo run --release -p tts_cli --no-default-features --features cuda -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --backend cuda \
  --output ./out/custom-voice-cuda-zh.wav
```

## CUDA Backend Note

The current CUDA path must avoid running the talker `codec_head` and
code-predictor `lm_head` directly on CUDA during greedy generation. That path
can leave the Burn/CubeCL CUDA runtime in an invalid launch state for this
model shape.

The stable workaround is:

- run the talker and code predictor up to readable hidden states
- synchronize those hidden states to host immediately inside the helper that
  produces them
- do greedy token selection on host from those hidden states
- rebuild fresh device-side integer token tensors for subsequent GPU stages

Keep the CUDA CLI check above passing before changing this generation path.

## Local Asset Assumptions

Current repo-local model directories include:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice`

CLI smoke verification assumes those paths exist locally and point at complete
runtime assets.

## Test Ownership Rules

### Framework core

Owns tests for:

- lifecycle state machine
- manager/remove/close semantics
- driver registration behavior
- diagnostics projection

### Qwen3 driver

Owns tests for:

- request and profile semantics
- load normalization
- execution chain correctness
- capability aggregation
- real-model synthesis smoke paths

### CLI

Owns tests for:

- command parsing
- shell-level output path behavior
- user-visible flag compatibility

## Acceptance Criteria

The testing document is acceptable only if it defines:

- which tests must remain fast and asset-free
- which tests are allowed to use real model assets
- where lifecycle tests belong
- where capability tests belong
- what the default verification command set is
- what must not remain inside CLI tests
