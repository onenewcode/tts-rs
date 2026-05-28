# TTS Inference Engine Architecture

The workspace is split into a reusable execution core and a Qwen-specific
model crate:

- `tts_core`: model-agnostic execution kernel, scheduling helpers, shared
  sampling runtime, and output abstractions.
- `tts_qwen`: Qwen-family runtime, release routing, request profiles, shared
  Qwen architectures, tokenizer loading, and WAV output helpers.

## Runtime Flow

```text
request -> tts_core runtime -> qwen release router -> qwen profile compiler -> qwen arch runner -> codec -> WAV
           ^                                                                 |
           |                                                                 v
    core session/events/chunking                                  backend + model lifecycle glue
```

## Public API

| API | Purpose |
|---|---|
| `tts_core::TtsService::synthesize()` | model-agnostic runtime entrypoint |
| `tts_qwen::register_qwen_family_model()` | register a Qwen release into the core registry |
| `tts_qwen::CustomVoiceRequest` | public request type for the exported custom-voice Qwen profile |

## Source Layout

```text
tts_core/src/
  executor.rs    core executor/run contract
  service.rs     core-owned runtime loop and chunking policy
  runtime/       shared sampling + KV primitives
  scheduler.rs   chunk emission scheduling helper

tts_qwen/src/
  arch/          shared qwen-family model structure, loaders, runners, and kernels
  profile/       request semantics, prompt rules, and request compilation
  releases.rs    family release metadata and release-to-architecture/profile routing
  runtime/       crate-local executor glue and runtime-local types
  io/            tokenizer and WAV helpers
  profiling/     profiling toggles and operator instrumentation
  registry.rs    qwen family registration entrypoint
```

## Design Rules

- Core orchestration lives in `tts_core`; model crates should not reimplement
  generic runtime loops, chunk scheduling, or event policy.
- Release selection is metadata-driven: a release resolves to one architecture
  runner plus one request profile.
- Architecture modules own model structure, weights, state, and forward paths.
- Profile modules own request semantics, prompt rules, and request compilation.
- Runtime modules own backend setup and run lifecycle; they must not absorb
  Qwen prompt or model-shape logic.
