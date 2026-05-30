# VibeVoice Integration Docs

## Summary

This directory captures the architecture and implementation plan for adding
VibeVoice into `tts-rs` as a new model family driver.

The current landing target is `VibeVoice-Realtime-0.5B`. The document set is
written for architecture and implementation work, not for end-user operation.

The guiding decision is:

- create a new crate named `tts_vibevoice`
- keep `tts_qwen3_tts` as a Qwen3-only driver
- allow short-term duplication where needed
- extract shared abstractions only after VibeVoice works inside the framework

## Reading Order

Read these documents in order:

1. [01-scope-and-driver-decision.md](01-scope-and-driver-decision.md)
2. [02-model-architecture-and-runtime.md](02-model-architecture-and-runtime.md)
3. [03-integration-stages.md](03-integration-stages.md)
4. [04-validation-and-risks.md](04-validation-and-risks.md)

Then use the appendices as needed:

- [appendix-a-cached-prompt-and-assets.md](appendix-a-cached-prompt-and-assets.md)
- [appendix-b-streaming-session-and-diffusion.md](appendix-b-streaming-session-and-diffusion.md)
- [appendix-c-future-variant-expansion.md](appendix-c-future-variant-expansion.md)

## Scope

This document set covers:

- why VibeVoice should land as `tts_vibevoice` instead of extending
  `tts_qwen3_tts`
- what is currently known about `VibeVoice-Realtime-0.5B`
- how its runtime differs from the existing Qwen3-TTS driver
- which staged implementation path best fits the current `tts-rs` architecture
- what expansion points should be reserved for future VibeVoice variants

This document set does not attempt to define a universal cross-model request
protocol.

## Evidence Base

Local repository sources used to ground the design:

- `docs/architecture.md`
- `tts_qwen3_tts/src/loading/mod.rs`
- `tts_qwen3_tts/src/loading/package/normalize.rs`
- `tts_qwen3_tts/src/execution/compiler/mod.rs`
- `tts_qwen3_tts/src/model/mod.rs`
- `tts_qwen3_tts/src/surface/request/voice_clone.rs`
- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/model.safetensors`

Official external references used to cross-check the local observations:

- VibeVoice repository: <https://github.com/microsoft/VibeVoice>
- Realtime docs: <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
- Realtime demo: <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
- Streaming inference implementation:
  <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming_inference.py>
- Streaming model implementation:
  <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming.py>
- Streaming processor implementation:
  <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/processor/vibevoice_streaming_processor.py>
- Text tokenizer implementation:
  <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modular_vibevoice_text_tokenizer.py>

## Relationship to the Main Architecture

These VibeVoice documents refine, but do not replace, the framework-level
architecture in `docs/architecture.md`.

At the framework layer, `tts-rs` standardizes:

- driver registration
- model loading
- loaded instance lifecycle
- serialized execution
- runtime capability projection

At the driver layer, `tts_vibevoice` will remain model-family specific.
