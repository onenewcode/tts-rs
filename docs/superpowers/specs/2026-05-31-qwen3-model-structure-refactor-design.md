# Qwen3 TTS Model Structure Refactor Design

## Goal

Refactor `tts_qwen3_tts/src/model` so it only contains model-private
implementation code. Move loaded-model runtime assembly, backend dispatch, and
session wiring into `tts_qwen3_tts/src/execution`.

## Scope

In scope:

- `tts_qwen3_tts/src/model`
- `tts_qwen3_tts/src/execution`
- `tts_qwen3_tts/src/loading/mod.rs`
- `tts_qwen3_tts/src/capabilities/mod.rs`

Out of scope:

- changing public crate API semantics
- splitting `tts_qwen3_tts` into multiple crates
- changing backend feature names or support policy

## Current Problems

### `model/` mixes implementation and runtime assembly

`tts_qwen3_tts/src/model/mod.rs` currently owns two unrelated responsibilities:

- model-private components such as `talker`, `speaker`, `codec`, and `nn`
- loaded-model lifecycle code such as backend selection, session start, and
  audio finalization

This makes `model/` mean both "neural model implementation" and "loaded model
runtime facade", which is a poor boundary and conflicts with the repository's
target layering.

### Backend dispatch is in the wrong layer

The backend-specific load path lives in `tts_qwen3_tts/src/model/runtime.rs`.
That logic is runtime orchestration, not model definition.

### Session and audio finalization are runtime concerns

Session stepping and audio finalization depend on the loaded model instance,
compiled request, and execution lifecycle. They are not part of the static
model component tree.

## Design Principles

1. `model/` means model-private implementation only.
2. `execution/` owns loaded-model runtime assembly and session lifecycle.
3. Internal paths may change freely; no compatibility layer is required for
   private modules.
4. Public behavior must remain unchanged.
5. Visibility should be reduced where practical; prefer private modules and
   `pub(crate)` items.

## Target Structure

```text
tts_qwen3_tts/src/
  execution/
    audio_finalize.rs
    backend_runtime.rs
    loaded_model.rs
    mod.rs
  model/
    codec/
    nn/
    speaker/
    talker/
    mod.rs
```

### `model/`

Responsibilities:

- talker config/network/inference/weights
- speaker encoder config/network/inference/weights
- codec config/network/inference/weights
- shared low-level tensor helpers used by model subsystems

Non-responsibilities:

- no loaded-model facade
- no backend dispatch
- no engine session wrapper

### Subtree alignment rule

`speaker/`, `codec/`, and `talker/` must follow the same structural vocabulary:

- `config.rs`
- `infer/`
- `network/`
- `weights.rs`

The intent is not identical file counts. The intent is identical responsibility
boundaries.

Rules:

- `config.rs` owns manifest/config types and model-construction entrypoints
- `weights.rs` owns safetensor/pytorch remapping and loaded wrapper structs
- `network/` owns neural network modules and architecture internals
- `infer/` owns runtime inference helpers and output conversion

Additional rule for large `network/` trees:

- when one `network/mod.rs` grows to hold multiple distinct operator families,
  split it by operator/subnetwork instead of keeping a single monolithic file

Implications:

- current `codec/model.rs` is a network concern and must move under
  `codec/network/`
- current `codec/runtime.rs` is an inference concern and must move under
  `codec/infer/`
- current `speaker/feature.rs` is part of inference preprocessing and must move
  under `speaker/infer/`
- current `talker/attention.rs`, `kv.rs`, `layer.rs`, `mlp.rs`, and `rope.rs`
  are network internals and must move under `talker/network/`
- `talker/sampling.rs` is inference-time behavior and must move under
  `talker/infer/`
- `codec/network/mod.rs` must not remain a single large file; it should be
  split into operator-focused modules such as encoder backbone, encoder
  transformer, encoder quantizer, decoder transformer, decoder quantizer, and
  wave decoder

### `execution/loaded_model.rs`

Responsibilities:

- define `Qwen3TtsLoadedModel`
- own `LoadedModelOps`, backend runtime wrapper, and session wrapper
- translate compiled requests into generator sessions

### `execution/backend_runtime.rs`

Responsibilities:

- select the backend implementation
- initialize wgpu-family devices where needed
- construct `Qwen3TtsModelInner` for the selected backend

### `execution/audio_finalize.rs`

Responsibilities:

- build codec prefix tensors for reference-audio continuation
- flatten reference codec frames into quantizer-major layout
- keep tests for those helpers close to the runtime boundary that uses them

## Symbol Migration

Move these symbols out of `model/mod.rs`:

- `Qwen3TtsModelInner`
- `LoadedModelOps`
- `BackendRuntime`
- `SessionOps`
- `Qwen3TtsLoadedModel`
- `Qwen3TtsSession`
- `SessionImpl`
- `start_backend_session`
- `map_sampling`
- `reference_codec_prefix_tensor`
- `flatten_reference_codec_frames`

Delete:

- `tts_qwen3_tts/src/model/runtime.rs`

Restructure:

- `tts_qwen3_tts/src/model/codec/model.rs` -> `codec/network/mod.rs`
- `tts_qwen3_tts/src/model/codec/runtime.rs` -> `codec/infer/mod.rs`
- `tts_qwen3_tts/src/model/speaker/feature.rs` -> `speaker/infer/feature.rs`
- `tts_qwen3_tts/src/model/speaker/infer.rs` -> `speaker/infer/mod.rs`
- `tts_qwen3_tts/src/model/speaker/network.rs` -> `speaker/network/mod.rs`
- `tts_qwen3_tts/src/model/talker/network.rs` -> `talker/network/mod.rs`
- `tts_qwen3_tts/src/model/talker/attention.rs` -> `talker/network/attention.rs`
- `tts_qwen3_tts/src/model/talker/kv.rs` -> `talker/network/kv.rs`
- `tts_qwen3_tts/src/model/talker/layer.rs` -> `talker/network/layer.rs`
- `tts_qwen3_tts/src/model/talker/mlp.rs` -> `talker/network/mlp.rs`
- `tts_qwen3_tts/src/model/talker/rope.rs` -> `talker/network/rope.rs`
- `tts_qwen3_tts/src/model/talker/infer.rs` -> `talker/infer/mod.rs`
- `tts_qwen3_tts/src/model/talker/sampling.rs` -> `talker/infer/sampling.rs`

## Dependency Direction

Allowed direction after refactor:

- `execution` -> `model`
- `loading` -> `execution`
- `capabilities` -> `execution`

Disallowed direction after refactor:

- `model` -> `execution` for loaded-model facade concerns

Exception:

- `Qwen3TtsModelInner::load` may still call execution profiling setup because
  profiling configuration is part of runtime load behavior and already exposed
  through the execution layer. This is acceptable until profiling is extracted
  further.

## Acceptance Criteria

The refactor is complete when all of the following are true:

- `tts_qwen3_tts/src/model/mod.rs` only declares model-private submodules
- `speaker/`, `codec/`, and `talker/` expose the same top-level responsibility
  split: config/infer/network/weights
- loaded-model/session/backend dispatch code is owned by `execution/`
- `tts_qwen3_tts/src/model/runtime.rs` no longer exists
- `loading/mod.rs` and `capabilities/mod.rs` depend on `execution` for
  `Qwen3TtsLoadedModel`
- crate tests and `clippy -D warnings` pass
