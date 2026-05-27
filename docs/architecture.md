# TTS Inference Engine Architecture

`tts_qwen` is now organized as an on-device inference engine instead of a one-shot
pipeline facade. The public entrypoint is `QwenTtsEngine`, which owns model
weights, request compilation, session state, scheduling, streaming, and
operator-level profiling.

## Runtime Flow

```text
request -> frontend compile -> talker prefill -> step decode -> codec decode -> audio events / WAV
                             ^                |
                             |                v
                         session state <-> scheduler
```

## Public API

| API | Purpose |
|---|---|
| `QwenTtsEngine::load()` | load weights, tokenizer, configs, profiling settings |
| `start_session()` | compile one request and allocate a session |
| `step()` | advance a session by one generation step |
| `drain_events()` | read streamed codec/audio events |
| `run_to_end()` | blocking helper that drives the session to completion |
| `finish_session()` | decode final waveform and release the session slot |

## Source Layout

```text
tts_qwen/src/
  engine/        public engine API and configuration
  session/       request types, session state, stream events
  scheduler/     low-latency single-session scheduling policy
  runtime/       KV cache and sampling primitives
  profiling/     operator timing configuration and logging helpers
  model/         configs, weight loading, model definitions
  runners/       frontend compile, talker generation, codec decode
  kernels/       hot-path tensor operators
  io/            tokenizer, model path helpers, WAV writing
```

## Design Rules

- Session state is explicit and long-lived; hot-path decode state is never hidden
  inside one-shot adapter functions.
- Streaming is event-driven; WAV writing is an edge concern layered on top.
- Operator profiling is optional and controlled by a crate feature plus runtime
  configuration.
- `core.rs`, `pipeline.rs`, and adapter-based orchestration were removed.
