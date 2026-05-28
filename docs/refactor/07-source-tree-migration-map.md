# Source Tree Migration Map

## Purpose

This document translates the refactor from architecture language into file-level
implementation moves. The goal is to let later Goal-mode coding proceed without
having to rediscover where each concept should land.

## Target Workspace Tree

```text
tts_infer/
  src/
    error.rs
    audio.rs
    engine.rs
    session.rs
    lib.rs

tts_qwen3_tts/
  src/
    backend.rs
    error.rs
    lib.rs
    package/
      manifest.rs
      normalize.rs
      mod.rs
    request/
      base.rs
      custom_voice.rs
      language.rs
      mod.rs
    profiling/
      config.rs
      mod.rs
    compiler/
      mod.rs
      prompt.rs
      profiles.rs
    model/
      mod.rs
      inner.rs
      loaded.rs
      session.rs
      seed.rs
    io/
      tokenizer.rs
      wav.rs
      mod.rs

tts_cli/
  src/
    cli.rs
    main.rs
    lib.rs
```

The exact filenames may shift slightly during implementation, but the ownership
boundaries must not.

## Legacy To Target Mapping

### `tts_core`

- `tts_core/src/types.rs` -> split and re-home only stable audio/output-facing
  pieces into `tts_infer/src/audio.rs`
- `tts_core/src/error.rs` -> do not preserve whole-cloth; replace with thin
  `tts_infer::ServiceError` and model-specific errors in `tts_qwen3_tts`
- `tts_core/src/service.rs` -> conceptually replaced by `tts_infer/src/engine.rs`
- `tts_core/src/executor.rs` -> delete; resident-model lifecycle now lives
  between `tts_infer` and `tts_qwen3_tts`
- `tts_core/src/registry.rs` -> delete outright
- `tts_core/src/scheduler.rs` -> delete unless a clearly reusable invariant
  survives; no placeholder port
- `tts_core/src/runtime/sampling.rs` -> v1 most likely moves to
  `tts_qwen3_tts/src/model` or `tts_qwen3_tts/src/compiler`; do not force it
  into `tts_infer`
- `tts_core/src/runtime/kv.rs` -> move only if still needed by backend-local
  model execution; new home should be inside `tts_qwen3_tts`
- `tts_core/src/wav.rs` -> either `tts_infer/src/audio.rs` or
  `tts_qwen3_tts/src/io/wav.rs`, depending on whether only `PcmAudio` writing
  remains after simplification

### `tts_qwen/src/profile/*`

- `tts_qwen/src/profile/base/request.rs` -> `tts_qwen3_tts/src/request/base.rs`
- `tts_qwen/src/profile/custom_voice/request.rs` ->
  `tts_qwen3_tts/src/request/custom_voice.rs`
- `tts_qwen/src/profile/base/prompt.rs` ->
  `tts_qwen3_tts/src/compiler/prompt.rs`
- `tts_qwen/src/profile/custom_voice/prompt.rs` ->
  `tts_qwen3_tts/src/compiler/prompt.rs`
- `tts_qwen/src/profile/base/config.rs` -> compiled-profile loading logic under
  `tts_qwen3_tts/src/compiler/profiles.rs`
- `tts_qwen/src/profile/custom_voice/config.rs` ->
  `tts_qwen3_tts/src/compiler/profiles.rs`
- `tts_qwen/src/profile/model_config.rs` -> package/compiler-facing config load
  helpers under `tts_qwen3_tts/src/package` or `tts_qwen3_tts/src/compiler`
- `tts_qwen/src/profile/compile.rs` -> absorbed into
  `Qwen3TtsRequestCompiler::compile_session_seed(...)`
- `tts_qwen/src/profile/mod.rs` -> replaced by `request/mod.rs` plus compiler
  modules

### `tts_qwen/src/arch/*`

Everything under `tts_qwen/src/arch` is deleted as a directory boundary.
Useful logic is re-homed by responsibility:

- `arch/engine/compiler.rs` -> `tts_qwen3_tts/src/compiler/mod.rs`
- `arch/engine/protocol.rs` -> deleted; replaced by `SessionSeed<B>`
- `arch/engine/spec.rs` -> delete unless a minimal config type is still needed
  by `model/inner.rs`
- `arch/engine/components/generator/*` -> mostly `tts_qwen3_tts/src/model/*`
- `arch/engine/components/decoder/*` -> `tts_qwen3_tts/src/model/*`
- `arch/kernels/*` -> backend/model-local modules inside `tts_qwen3_tts`
- `arch/engine/mod.rs` and `arch/mod.rs` -> deleted, not renamed

### `tts_qwen/src/runtime/*`

- `tts_qwen/src/runtime/executor.rs` -> split between `backend.rs`,
  `model/loaded.rs`, and `model/session.rs`
- `tts_qwen/src/runtime/types.rs` -> absorbed into target request/run-option
  types in `lib.rs`, `request/*`, or `profiling/config.rs`
- `tts_qwen/src/runtime/mod.rs` -> deleted

### `tts_qwen/src/releases.rs` and `tts_qwen/src/registry.rs`

- `tts_qwen/src/releases.rs` -> delete public release layer; keep only concrete
  package facts under `package/*`
- `tts_qwen/src/registry.rs` -> delete outright; no replacement

### `tts_qwen/src/backend.rs`

- replace with the single authoritative `Qwen3TtsBackend` enum and its parsing,
  availability, and resolution helpers in `tts_qwen3_tts/src/backend.rs`

### `tts_qwen/src/io/*`

- `tts_qwen/src/io/tokenizer.rs` -> `tts_qwen3_tts/src/io/tokenizer.rs`
- `tts_qwen/src/io/wav.rs` -> keep only if model crate still needs a private WAV
  helper; otherwise push audio writing beside `PcmAudio`

### `tts_qwen/src/profiling/*`

- `tts_qwen/src/profiling/config.rs` ->
  `tts_qwen3_tts/src/profiling/config.rs`
- `tts_qwen/src/profiling/mod.rs` -> `tts_qwen3_tts/src/profiling/mod.rs`

### `tts_cli/*`

- `tts_cli/src/cli.rs` -> keep file, but replace old `models.yaml` and
  `variant` argument surface with package-first subcommands
- `tts_cli/src/main.rs` -> remains thin
- `tts_cli/src/lib.rs` -> remains thin or disappears if it provides no value

## Delete-Only Areas

The following concepts should not receive compatibility wrappers:

- `ModelRegistry`
- `SynthesisRequest` shared across families
- public release manifests
- public variant labels as the main selection handle
- old executor/family registration path
- the `arch` directory boundary itself

## Implementation Order Constraint

The recommended coding order for the later Goal-mode run is:

1. create `tts_infer` with session contract and tests
2. create `tts_qwen3_tts` request/package/profiling types
3. implement package normalization and compiler profile loading
4. port resident model load/session execution
5. rewrite `tts_cli` against package-first API
6. delete `tts_core` and old `tts_qwen`

This is not a product roadmap. It is a dependency-safe file migration order.
