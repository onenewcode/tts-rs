# Testing Strategy For TTS-RS

## Summary

Testing must follow the target architecture, not only the current file layout.

The repository has three current crates, but the test strategy should already
enforce future boundaries:

- framework-core behavior stays fast and deterministic
- Qwen3 driver behavior is split between public-surface tests and deeper driver
  tests
- model-backed smoke paths stay ignored and explicitly documented
- CLI tests only verify shell behavior and argument parsing

## Test Layers

### 1. Core fast tests

Current location:

- `tts_infer/tests/`

Future location:

- `tts_core/tests/`

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

### 4. Real-model ignored smoke tests

Current location:

- `tts_qwen3_tts/tests/real_model.rs`

Must verify:

- custom-voice synthesis
- instructed synthesis
- base voice-clone synthesis
- x-vector-only voice-clone synthesis

Rules:

- keep these tests `#[ignore]`
- do not require them in default CI-style fast checks
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
cargo test -p tts_infer
cargo test -p tts_qwen3_tts --test public_surface
cargo test -p tts_qwen3_tts --test compiler_load
cargo test -p tts_cli --test cli_parse
```

Current repo note:

- `cargo test -p tts_infer` is already passing in the observed repository state

Recommended workspace fast path after refactor:

```bash
cargo test --workspace -- --skip real_model
```

Model-backed smoke path:

```bash
cargo test -p tts_qwen3_tts --test real_model -- --ignored --nocapture
```

## Local Asset Assumptions

Current repo-local model directories include:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice`

Real-model smoke tests must tolerate absent local assets by returning early,
but documentation should state the expected paths and required files.

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

