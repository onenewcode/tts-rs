# Migration And Acceptance

## Major Structural Moves

### Delete

- `tts_core`
- `tts_qwen/src/arch`
- old public release/variant selection APIs
- old registry-based family registration
- old fake-streaming/event loop contract
- duplicated backend enums across crates

### Create

- `tts_infer`
- `tts_qwen3_tts`
- thin `tts_cli` over the new model crate

## Migration Mapping

### Legacy `tts_core`

Delete the crate, but selectively re-home useful implementation pieces:

- shared PCM/WAV writing logic may move to `tts_infer` or `tts_qwen3_tts`
- sampling utilities likely move under `tts_qwen3_tts` in v1 unless a clearly
  stable seam appears during implementation
- legacy generic request/service/registry abstractions must not be preserved

### Legacy `tts_qwen` compiler path

The old chain:

- request
- prepared condition
- execution form
- compiled request
- run

must be flattened into:

- request
- `compile_session_seed()`
- `start_generator()`
- session `step()` / `finish()`

### Legacy backend glue

The old executor path around `QwenBackendRun` is conceptually reused but moved
into the model crate's own loaded-model/session backend erasure:

- `Qwen3TtsLoadedModel` for resident model erasure
- `Qwen3TtsSession` for run-time session erasure

### Legacy prompt/profile modules

Prompt and profile logic remain in the model crate but are reorganized around:

- request structs
- compiled profiles
- prompt recipe enum
- request compiler

No new `arch`-style hierarchy should reappear.

## Acceptance Criteria

Implementation is not complete until all of the following are true:

- workspace only contains `tts_infer`, `tts_qwen3_tts`, and `tts_cli`
- `tts_core` is removed
- `tts_qwen/src/arch` is removed
- `Qwen3TtsEngine::load(...)` is the public load entry
- `Qwen3TtsEngine::synthesize(...)` is the public one-shot synthesis entry
- `tts_infer` owns the internal session state machine and `PcmAudio`
- `tts_qwen3_tts` owns request semantics, package parsing, backend selection,
  compiler logic, and model execution
- public variant/release selection is gone
- CLI is package-first and profile-subcommand driven

## Validation Targets

### Architecture Validation

- no crate named `tts_core` remains in workspace membership
- no `src/arch` tree remains under the model crate
- no shared fake-generic request type exists in `tts_infer`
- backend has one true enum in `tts_qwen3_tts`

### API Validation

- `Qwen3TtsEngine::load(Qwen3TtsEngineConfig)` compiles and loads a package
- `Qwen3TtsEngine::synthesize(QwenRequest, Qwen3TtsRunOptions)` returns
  `PcmAudio`
- `BaseRequest` rejects speaker semantics at compile/validation path
- `CustomVoiceRequest` resolves `speaker` and `language` through package facts

### Model-Backed Validation

- package manifest resolves all required assets
- compiler loads profile config once during engine load
- synthesis produces mono, 24 kHz, 16-bit PCM output
- the CLI smoke path goes through package-first load and profile subcommands

### Session Contract Validation

- `synthesize()` is implemented as `start_session + loop + finish`
- terminal session state forbids further `step()`
- pre-terminal `finish()` returns service-layer invariant error
- model-layer errors remain wrapped in `InferError::Model(...)`

## Testing Document Direction

When code starts landing, `docs/testing_tts_qwen.md` should be rewritten to
track the new crate names and API paths. It should stop referencing:

- `tts_core`
- release/variant routing tests
- `tts_qwen/src/arch`
- fake-streaming chunk policy tests
