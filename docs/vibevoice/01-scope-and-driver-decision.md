# Scope and Driver Decision

## Purpose

This document records the architecture decision for how VibeVoice should enter
`tts-rs`.

The main question is whether VibeVoice should:

- extend `tts_qwen3_tts`
- land as a new sibling crate
- or first become a sibling crate and only later contribute shared abstractions

The recommended decision is the third option.

## Current Framework Context

`docs/architecture.md` defines the framework center of gravity as a local model
runtime, not a universal synthesis request protocol. That framing matters here:
VibeVoice is free to expose very different runtime semantics as long as it fits
loaded-instance lifecycle, capability projection, and serialized execution.

The existing Qwen driver already reflects this model-private structure:

- `tts_qwen3_tts/src/loading/mod.rs` loads package, compiler, model, and then
  projects capabilities
- `tts_qwen3_tts/src/loading/package/normalize.rs` expects a Qwen-specific
  directory contract
- `tts_qwen3_tts/src/model/mod.rs` assumes a talker plus codec decoder runtime
- `tts_qwen3_tts/src/surface/request/voice_clone.rs` exposes Qwen-specific
  voice-clone prompt semantics

Those assumptions are not shared by VibeVoice-Realtime.

## Observed Facts About VibeVoice-Realtime

From the local model directory and upstream docs:

- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md` describes an interleaved,
  windowed, realtime TTS model built around diffusion-based acoustic latent
  generation
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json` declares a streaming
  architecture with a Qwen2-based decoder stack, a dedicated diffusion head,
  and an acoustic tokenizer config
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json` points to an
  external base tokenizer source, `Qwen/Qwen2.5-0.5B`, instead of shipping a
  local tokenizer file beside the model weights
- the local `model.safetensors` names show modules such as
  `model.language_model.*`, `model.tts_language_model.*`,
  `model.prediction_head.*`, `model.acoustic_connector.*`, and
  `model.acoustic_tokenizer.decoder.*`
- the official realtime demo loads cached voice prompts from `.pt` files rather
  than building a prompt from arbitrary user reference audio at request time

These facts point to a runtime that differs materially from Qwen3-TTS.

## Option Analysis

### Option A: Extend `tts_qwen3_tts`

Advantages:

- least short-term crate scaffolding
- immediate reuse of some familiar folder names

Costs:

- package loading rules become mixed between Qwen-specific and VibeVoice-
  specific asset layouts
- request types become harder to reason about because Qwen base/custom-voice
  semantics do not match cached-prompt realtime semantics
- the runtime would need to mix two incompatible generation models: discrete
  codec-token autoregression and diffusion-based continuous latent generation
- the crate name would stop describing its contents accurately

Conclusion: not recommended.

### Option B: Create a separate sibling crate and keep everything separate

Advantages:

- clean naming and strong ownership boundaries
- easier capability projection and testing
- lower risk of contaminating Qwen-specific code paths

Costs:

- some boilerplate will be duplicated even where the structure is similar
- no immediate path for extracting stable shared abstractions

Conclusion: better than Option A, but leaves long-term cleanup unspecified.

### Option C: Create `tts_vibevoice` now, then extract shared abstractions later

Advantages:

- preserves clean driver ownership boundaries now
- allows short-term duplication where necessary
- defers abstraction work until VibeVoice realities are known from an actual
  working driver
- aligns with `docs/architecture.md`, which centers framework lifecycle rather
  than input unification

Costs:

- requires discipline to avoid premature refactoring into `tts_core`
- may temporarily duplicate loading/session wrapper patterns

Conclusion: recommended.

## Decision

Create a new sibling crate named `tts_vibevoice`.

For the first implementation cycle:

- `tts_qwen3_tts` remains Qwen3-only
- `tts_vibevoice` becomes the VibeVoice-family landing zone
- framework-level reuse is limited to already-stable `tts_core` lifecycle and
  capability mechanisms
- any newly discovered common abstractions stay private until at least one
  VibeVoice path is working and understood

## Why `tts_vibevoice` Fits the Repository Better

This choice preserves the intended layering already described in the repo:

- `tts_core` owns common local model-service concerns
- driver crates own model-private loading, request semantics, and runtime logic
- `tts_app` owns orchestration
- `tts_cli` remains a thin shell

Under that architecture, VibeVoice belongs beside Qwen, not inside it.

## Success Criteria for the First VibeVoice Landing

The first VibeVoice landing is successful when all of the following are true:

- `tts_vibevoice` exists as a separate workspace crate
- the crate can describe and validate a VibeVoice-Realtime asset package
- the crate can expose a driver-specific request surface without borrowing
  Qwen-only concepts
- the crate can project correct runtime capabilities into `tts_core`
- the crate has a clear staged path to minimal end-to-end verification through
  `tts_app` and `tts_cli`

## References

Local:

- `docs/architecture.md`
- `tts_qwen3_tts/src/loading/mod.rs`
- `tts_qwen3_tts/src/loading/package/normalize.rs`
- `tts_qwen3_tts/src/model/mod.rs`
- `tts_qwen3_tts/src/surface/request/voice_clone.rs`
- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`

Official:

- <https://github.com/microsoft/VibeVoice>
- <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
- <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
