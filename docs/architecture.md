# TTS Inference Engine Architecture

The workspace is split into a reusable execution core and a model-family
implementation:

- `tts_core`: model-agnostic execution kernel, scheduling helpers, shared
  sampling runtime, and output abstractions.
- `tts_qwen`: Qwen-family executor implementation (weights, kernels, request
  compilation, talker forward, and codec decode).

## Runtime Flow

```text
 request -> tts_core runtime -> qwen executor -> talker prefill/decode -> codec decode -> WAV
            ^                               |
            |                               v
     core session/events/chunking      qwen kernels
```

## Public API

| API | Purpose |
|---|---|
| `tts_core::TtsService::synthesize()` | model-agnostic runtime entrypoint |
| `tts_qwen::register_qwen_family_model()` | register the Qwen family executor into the core registry |
| `tts_qwen::engine::QwenTtsEngine` | internal loaded model bundle for the Qwen executor |

## Source Layout

```text
tts_core/src/
  executor.rs    core executor/run contract
  service.rs     core-owned runtime loop and chunking policy
  runtime/       shared sampling + KV primitives
  scheduler.rs   chunk emission scheduling helper

tts_qwen/src/
  executor.rs    qwen family executor and backend dispatch
  engine/        loaded qwen model bundle + per-request run state
  frontend.rs    qwen request types, prompt/config loading, and request compilation
  model/         configs, weights, model definitions
  runners/       talker generation and codec decode
  kernels/       hot-path tensor operators
  io/            tokenizer and WAV helpers
```

## Design Rules

- Core orchestration lives in `tts_core`; model crates should not reimplement
  generic runtime loops, chunk scheduling, or event policy.
- Qwen-side code is limited to request compilation, model forward, and audio
  decode for the Qwen family.
- Runtime-facing state machines stay in core; family crates own only the
  minimal per-request model state they need to advance inference.
