# TTS Inference Engine Architecture

The workspace is split into a reusable execution core and a model-family
implementation:

- `tts_core`: model-agnostic service/session orchestration, scheduling helpers,
  shared sampling runtime, and output abstractions.
- `tts_qwen`: Qwen-family inference implementation (weights, kernels, codec,
  and provider glue).

## Runtime Flow

```text
request -> tts_core service -> qwen provider -> talker prefill/decode -> codec decode -> WAV
            ^                              |
            |                              v
      core sampling/scheduler/kv     qwen kernels
```

## Public API

| API | Purpose |
|---|---|
| `tts_core::TtsService::synthesize()` | model-agnostic orchestration entrypoint |
| `tts_qwen::QwenFamilyAdapter` | Qwen family provider implementation |
| `QwenTtsEngine::load()` | internal engine load for qwen provider sessions |
| `QwenTtsEngine::{start_session,step,drain_events,finish_session}` | provider-facing incremental execution API |

## Source Layout

```text
tts_core/src/
  service.rs     model-agnostic orchestration entrypoint
  runtime/       shared sampling + KV primitives
  scheduler.rs   chunk emission scheduling helper

tts_qwen/src/
  provider.rs    qwen family adapter that binds core to qwen engine
  engine/        qwen incremental engine
  pipeline/      qwen request/session data contracts
  model/         configs, weights, model definitions
  runners/       frontend compile, talker generation, codec decode
  kernels/       hot-path tensor operators
  io/            tokenizer and WAV helpers
```

## Design Rules

- Core orchestration lives in `tts_core`; model crates should not reimplement
  generic service loops or sampling logic.
- Qwen-side code focuses on inference-only responsibilities.
- Streaming remains event-driven; WAV writing is an edge concern.
