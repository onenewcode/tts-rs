# Testing TTS-RS

## Summary

The repository has both fast, asset-free tests and model-backed smoke paths.
Testing should reinforce the current workspace boundaries:

- framework behavior stays fast and deterministic
- application-service request preparation stays outside the CLI shell
- Qwen3 public-surface and loading behavior are tested without real weights
- CLI parsing stays separate from model execution
- end-to-end synthesis runs stay explicit and release-mode only

## Recommended Fast Checks

Run these first when making ordinary code changes:

```bash
cargo test --release -p tts_infer
cargo test --release -p tts_app --test service
cargo test --release -p tts_qwen3_tts
cargo test --release -p tts_qwen3_tts --test compiler_load
cargo test --release -p tts_cli --test cli_parse
```

These commands match the current workspace members and test targets:

- `tts_infer/tests/audio.rs`
- `tts_infer/tests/manager.rs`
- `tts_app/tests/service.rs`
- `tts_qwen3_tts/src/` unit tests
- `tts_qwen3_tts/tests/compiler_load.rs`
- `tts_cli/tests/cli_parse.rs`

## Test Layers

### 1. Framework fast tests

Current location:

- `tts_infer/tests/`

Focus:

- loaded-model lifecycle rules
- manager/remove/close semantics
- driver registration behavior
- audio serialization primitives

These tests should stay fast, deterministic, and asset-free.

### 2. Application-service tests

Current location:

- `tts_app/tests/service.rs`

Focus:

- moving shell semantics out of the CLI layer
- request preparation for base and custom-voice flows
- input validation

These tests should stay asset-free.

### 3. Qwen3 public-surface tests

Current location:

- unit tests under `tts_qwen3_tts/src/`
- `tts_qwen3_tts/tests/compiler_load.rs`

Focus:

- public request defaults
- package source normalization
- public load/config defaults
- driver-facing request compilation/load boundaries

These tests should avoid real model weights and large tensor execution.

### 4. Qwen3 internal tests

Current location:

- unit tests under `tts_qwen3_tts/src/`

Focus:

- request pre-processing invariants
- execution-stage contracts
- capability aggregation rules
- backend boundary behavior
- model-private helper logic

Keep these tests near the code they verify.

### 5. CLI shell tests

Current location:

- `tts_cli/tests/cli_parse.rs`
- unit tests inside `tts_cli/src/cli.rs`

Focus:

- clap parsing
- command shape
- shell-level flag compatibility

These tests should not verify model internals or full synthesis semantics.

## CLI End-to-End Smoke Tests

End-to-end verification should use the CLI in release mode against local model
assets. A default custom-voice smoke path is:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --output ./out/custom-voice-flex-smoke.wav
```

A base-model voice-clone smoke path is:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone ICL smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --ref-text "Hello from the generated reference clip." \
  --output ./out/base_clone_icl_release.wav
```

These paths exercise CLI parsing, `tts_app` request preparation, model loading,
generation, and WAV writing together.

Runtime dtype conversion is selected with one CLI flag:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --dtype f32 \
  --output ./out/custom-voice-f32-smoke.wav
```

Supported values are `f32` and `bf16`. The loader converts float weights
directly into the requested runtime dtype during `load_from`, so the resident
model already uses the target precision when load returns. `f16` is rejected
explicitly because the current synthesis path does not produce correct output
with it. When `--dtype` is omitted, the CLI and driver default to `bf16`.

Rules:

- keep smoke runs out of the default fast test loop
- use release builds for realistic runtime behavior
- document any local asset assumptions alongside the command you run

## Optional Backend Verification

CUDA CLI verification:

```bash
cargo run --release -p tts_cli --no-default-features --features cuda -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --output ./out/custom-voice-cuda-zh.wav
```

### CUDA backend note

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
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

Smoke verification assumes those paths exist locally and point at complete
runtime assets.

## Ownership Guidelines

### `tts_infer`

Owns tests for:

- lifecycle state transitions
- manager/remove/close semantics
- driver registration behavior
- shared capability/result primitives

### `tts_app`

Owns tests for:

- request preparation
- shell-to-driver translation rules
- service-level validation and defaults

### `tts_qwen3_tts`

Owns tests for:

- public request/load behavior
- execution chain correctness
- capability aggregation
- backend-sensitive behavior
- real-model smoke procedures

### `tts_cli`

Owns tests for:

- command parsing
- shell-level output-path and flag behavior
- user-visible CLI compatibility

## Acceptance Criteria

The testing guide is complete when it answers:

- which tests are fast and asset-free
- which checks use real model assets
- how to run a CLI end-to-end smoke path
- where framework, app, driver, and CLI tests belong
- which commands are the default verification set
