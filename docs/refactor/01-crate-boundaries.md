# Crate Boundaries

## Workspace Target

The target workspace members are:

```toml
[workspace]
members = [
    "tts_infer",
    "tts_qwen3_tts",
    "tts_cli",
]
```

## Crate Responsibilities

### `tts_infer`

Owns only stable inference-service concerns:

- `PcmAudio`
- internal `LoadedModel` / `ModelSession` traits
- `Engine<M>`
- `EngineSession<S>`
- tiny orchestration errors and session-state guards

Must not own:

- model package schema
- prompt/profile semantics
- backend registry beyond what a concrete model hides
- generic TTS request structures
- tokenizer/config/model-specific lowering logic

### `tts_qwen3_tts`

Owns all model knowledge for the current Qwen3-TTS model:

- public `Qwen3TtsEngine`
- request types and request validation
- package manifest and package parsing
- backend enum and load-time backend resolution
- request compiler
- loaded-model resident state
- model session state
- final audio generation from model outputs

### `tts_cli`

Owns only entrypoint concerns:

- package-first CLI parsing
- profile subcommands
- log setup
- mapping CLI inputs to `QwenRequest` and `Qwen3TtsRunOptions`
- writing `PcmAudio` to WAV

Must remain a thin adapter over `tts_qwen3_tts`.

## Explicit Deletions

The implementation should remove, not preserve behind compatibility shims:

- the `tts_core` crate
- `tts_qwen/src/arch`
- the old release/variant public layer
- old registry-based model registration APIs
- shared fake-generic request structs
