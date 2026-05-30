# Integration Stages

## Purpose

This document turns the VibeVoice design discussion into a staged implementation
plan for `tts_vibevoice`.

Each stage is written as an architecture and implementation checkpoint:

- goal
- current facts
- required decisions
- recommended direction
- completion signal

The stages are ordered to reduce redesign churn.

## Stage 0: Preconditions and Non-Goals

### Goal

Define what the first VibeVoice landing is trying to accomplish.

### Current facts

- the first concrete target is `VibeVoice-Realtime-0.5B`
- the framework already supports driver registration and loaded-instance
  lifecycle through `tts_core`
- `tts_qwen3_tts` is a useful reference, but it reflects a different runtime
  model

### Required decisions

- current target variant
- what not to solve in the first pass
- whether cross-model request unification is in scope

### Recommended direction

For the first implementation cycle:

- target only `VibeVoice-Realtime-0.5B`
- do not redesign `tts_core` around VibeVoice before a working driver exists
- do not attempt to solve all future VibeVoice variants in code up front
- do not extend `tts_qwen3_tts` as the hosting crate

### Completion signal

This stage is complete when the implementation effort is explicitly framed as:

- a new crate `tts_vibevoice`
- a realtime cached-prompt first landing
- a framework-hosted driver, not a framework redesign

## Stage 1: Establish `tts_vibevoice` crate boundaries

### Goal

Create a crate boundary that matches VibeVoice responsibilities.

### Current facts

- the repo already separates framework concerns from Qwen driver concerns
- VibeVoice requires distinct loading, request, and runtime semantics

### Required decisions

- crate name and placement
- module split inside the crate
- dependency direction to `tts_core`, `tts_error`, `tts_app`, and `tts_cli`

### Recommended direction

Create `tts_vibevoice` as a sibling crate with internal modules analogous in
scope, but not in semantics, to the current Qwen driver:

- `surface/`
- `loading/`
- `capabilities/`
- `execution/`
- `model/`
- `backend/`
- `error.rs`
- `lib.rs`

The crate should:

- depend on `tts_core` for framework contracts
- depend on `tts_error` for diagnostics style
- avoid direct CLI concerns
- expose a thin public driver surface suitable for `tts_app`

### Completion signal

This stage is complete when the crate boundary is documented and accepted as the
single landing zone for VibeVoice-family work.

## Stage 2: Define model assets and loading protocol

### Goal

Translate the published VibeVoice-Realtime asset layout into a stable internal
package contract.

### Current facts

- the local model directory includes `config.json`, `configuration.json`,
  `preprocessor_config.json`, and `model.safetensors`
- the local model directory does not include a local tokenizer file comparable
  to the one expected by `tts_qwen3_tts`
- cached voice prompts are separate `.pt` assets in the upstream demo flow
- the processor references an external pretrained language-model tokenizer

### Required decisions

- what counts as a valid model directory
- whether prompt assets belong to package loading or request-time input
- how to represent external tokenizer and processor requirements
- whether to support both direct model-directory loading and manifest-driven
  loading

### Recommended direction

Support both:

- `ModelDir(PathBuf)` for the most natural upstream layout
- `ManifestPath(PathBuf)` for reproducible local packaging if needed later

Treat cached prompt assets as request-time inputs, not as required model-package
artifacts. The package describes the model runtime; the request supplies the
chosen voice prompt.

The first package contract should validate at least:

- `config.json`
- `configuration.json`
- `preprocessor_config.json`
- `model.safetensors`

The loading design should also preserve explicit metadata for:

- external tokenizer source
- expected processor class
- sample rate and compression ratio
- architecture kind

### Completion signal

This stage is complete when `tts_vibevoice` has a written package contract that
can explain missing assets and unsupported layouts with precise diagnostics.

## Stage 3: Define surface request and run options

### Goal

Expose a request model that matches VibeVoice-Realtime without prematurely
committing to family-wide abstractions.

### Current facts

- the upstream realtime flow requires text plus a cached prompt
- generation also accepts tuning such as CFG scale and DDPM inference step count
- the driver is single-speaker and English-first in the published release

### Required decisions

- what the first request struct must contain
- what belongs in engine config versus per-run options
- whether to expose streaming behavior in the first public surface

### Recommended direction

The first request surface should revolve around:

- synthesis text
- cached prompt source or prepared prompt object reference
- optional stop or length policy hooks only if they are needed immediately

The first run options should cover:

- CFG scale
- diffusion inference steps, if runtime-tunable after load
- verbosity or profiling hooks only if they map to an actual framework need

The first engine config should cover:

- model package source
- backend selection
- profiling and runtime behavior that must be fixed at load time

Do not expose a fake voice-clone API that implies arbitrary reference-audio
support if the published release does not support that path cleanly.

### Completion signal

This stage is complete when the driver surface is small, VibeVoice-specific, and
free of Qwen-only terminology.

## Stage 4: Define loaded instance and resident runtime structure

### Goal

Define what stays resident after loading and what is created per session.

### Current facts

- the upstream runtime uses model, processor, and cached prefilled state
  differently
- not all state belongs at the same lifetime
- framework handles loaded instance lifetime separately from request execution

### Required decisions

- which objects are instance-level
- which objects are session-level
- how cached prompt data enters the runtime
- how to classify and report load versus run errors

### Recommended direction

Instance-level state should include:

- validated package description
- instantiated backend/model runtime
- processor or equivalent request preparation component
- capability projection state
- backend/profiling configuration

Session-level state should include:

- prepared text inputs
- prompt-specific prefilled outputs
- per-run LM and TTS-LM cache state
- diffusion scheduler state if it must be isolated per request
- acoustic decoder streaming cache state

Error boundaries should distinguish:

- package/asset layout errors
- processor/config incompatibility errors
- cached prompt shape or contract errors
- runtime stepping failures
- finish/finalization failures

### Completion signal

This stage is complete when the loaded-instance design has a clear lifetime map
and no request-specific state is accidentally promoted to resident global state.

## Stage 5: Define the streaming session state machine

### Goal

Model VibeVoice-Realtime as an explicit session instead of hiding it behind a
single opaque synthesize call.

### Current facts

- upstream generation uses interleaved text-window and speech-window progress
- the implementation defines text and speech window sizes
- EOS classification is part of the runtime
- audio is produced incrementally

### Required decisions

- what states the session exposes internally
- how chunk emission maps to framework behavior
- what terminates generation
- whether the first public framework surface emits final audio only or also
  supports streaming callbacks

### Recommended direction

The internal session design should distinguish at least:

- prompt prefill ready
- text window advance
- speech window generation
- terminal reached
- finalization

The first public framework path may still choose to accumulate audio into a
final `PcmAudio` result, but the internal state machine should preserve the
realtime chunking model so that future streaming APIs are not blocked by an
incorrect first implementation.

### Completion signal

This stage is complete when the session can be described and tested as a real
state machine, not just as a monolithic loop.

## Stage 6: Connect `tts_app` and `tts_cli` through a minimal user path

### Goal

Define the smallest top-to-bottom product path for trying the VibeVoice driver
inside this repository.

### Current facts

- `tts_app` already owns orchestration responsibilities
- `tts_cli` should remain thin
- VibeVoice requires a cached prompt path in normal operation

### Required decisions

- whether the first user path is hidden/internal or fully CLI-exposed
- how a cached prompt path is supplied to the application layer
- which parameters are surfaced immediately and which remain fixed defaults

### Recommended direction

The first end-to-end path should be conservative:

- let `tts_app` own driver selection and argument translation
- keep `tts_cli` thin and explicit
- expose the minimum extra VibeVoice arguments needed to exercise the driver
  honestly, especially cached prompt input
- prefer documented defaults for experimental runtime knobs over a large first
  CLI surface

### Completion signal

This stage is complete when there is a documented minimal invocation path for a
VibeVoice request inside `tts-rs`, even if the user-facing CLI surface remains
intentionally narrow.

## Stage 7: Validate first, then extract shared abstractions

### Goal

Avoid abstracting over VibeVoice before the driver has proven which structure is
actually shared.

### Current facts

- some wrapper patterns will resemble `tts_qwen3_tts`
- the core runtime semantics are still materially different
- premature extraction would likely bake Qwen assumptions into shared code or do
  the reverse

### Required decisions

- which duplication is acceptable short-term
- what criteria justify extraction into `tts_core` or another shared module

### Recommended direction

Short-term duplication is acceptable for:

- package loading wrappers
- thin engine/handle glue
- capability projection glue
- model-private error enums that later reveal a shared subset

Shared extraction should occur only when:

- at least two drivers need the same abstraction
- the abstraction can be described without model-family leakage
- the extracted API improves correctness, not just aesthetics

### Completion signal

This stage is complete when the repo has a working VibeVoice path and a small,
credible list of candidate abstractions backed by actual implementation overlap.

## References

- `docs/architecture.md`
- `tts_qwen3_tts/src/loading/mod.rs`
- `tts_qwen3_tts/src/loading/package/normalize.rs`
- `tts_qwen3_tts/src/execution/compiler/mod.rs`
- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`
- <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming_inference.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/processor/vibevoice_streaming_processor.py>
