# Validation and Risks

## Purpose

This document defines how the future `tts_vibevoice` effort should be validated
and what the main risks are before and during implementation.

The emphasis is evidence-first verification rather than optimistic design.

## Validation Layers

### 1. Static package validation

Goal:

- prove that the local VibeVoice package contract is recognized correctly

Checks:

- required files are present
- config files parse successfully
- architecture kind matches the expected streaming runtime
- processor metadata is captured
- sample-rate and compression metadata are visible to the loader

Failure examples:

- missing `preprocessor_config.json`
- unsupported processor class
- unknown architecture label
- model directory that looks like a Qwen package rather than a VibeVoice package

### 2. Prompt-asset contract validation

Goal:

- prove that cached prompt assets can be validated before a full run begins

Checks:

- prompt object loads successfully
- required sub-structures exist
- prompt tensors or metadata match the expected runtime shape family
- prompt asset model family matches the loaded runtime

Failure examples:

- wrong prompt file type
- prompt built for a different variant
- missing expected prefilled outputs such as LM or TTS-LM state

### 3. Minimal runtime load validation

Goal:

- prove that a `tts_vibevoice` instance can load all mandatory resident state

Checks:

- backend creation succeeds
- processor/tokenizer setup succeeds
- model runtime and scheduler setup succeed
- capabilities can be projected after load

Failure examples:

- missing external tokenizer dependency
- backend unsupported on current machine
- incompatible config and model weight layout

### 4. Minimal single-request synthesis validation

Goal:

- prove the driver can execute one honest request with cached prompt input

Checks:

- request preparation works
- prompt prefill is accepted
- text and speech window advancement occurs
- final audio is returned with the correct sample rate and channel count

Failure examples:

- session fails during prefill
- diffusion loop fails mid-run
- decoder cache logic breaks finalization

### 5. Long-run or multi-window validation

Goal:

- prove the driver handles the architecture it claims to support, not just a
  tiny smoke case

Checks:

- multiple text windows succeed
- speech window stepping remains stable
- stop conditions and EOS behavior are correct
- long-form memory behavior is acceptable for local use

Failure examples:

- success only on short prompts
- accumulated cache corruption
- finalization only works when a single window is generated

## Main Risks

### Risk 1: The local model directory is not self-contained

Evidence:

- `preprocessor_config.json` points to `Qwen/Qwen2.5-0.5B`
- the local model directory does not mirror the Qwen package layout used by
  `tts_qwen3_tts`

Impact:

- loading may depend on external tokenizer/runtime code beyond the local folder
- local offline assumptions may fail unless the package contract is explicit

Mitigation:

- treat tokenizer/processor provenance as first-class loader metadata
- document unsupported offline layouts early

### Risk 2: Cached prompt assets are a critical runtime dependency

Evidence:

- upstream realtime demo and processor are prompt-centric
- local model release does not present arbitrary reference-audio cloning as the
  primary public flow

Impact:

- the first user path will fail unless prompt asset handling is designed
  explicitly

Mitigation:

- treat cached prompt validation as an independent test layer
- do not hide prompt requirements behind generic voice-clone naming

### Risk 3: Upstream runtime semantics are more streaming-native than the
first `tts-rs` surface may be

Evidence:

- upstream implementation advances through text and speech windows
- audio is produced incrementally

Impact:

- a simplistic one-shot wrapper may paint the implementation into a corner
- future streaming support could require a rewrite if the internal state machine
  is flattened too aggressively

Mitigation:

- preserve a session-state model internally even if the first public surface
  returns final audio only

### Risk 4: Family-wide VibeVoice abstraction is underspecified

Evidence:

- the repo currently has concrete access to `Realtime-0.5B`
- the official repository history for other TTS variants is not a stable basis
  for immediate code planning

Impact:

- over-generalized early abstractions could be wrong

Mitigation:

- scope the first code path tightly to `Realtime-0.5B`
- reserve future extension points in docs and type names without implementing
  speculative behavior immediately

### Risk 5: Backend reality may diverge from the rest of the workspace

Evidence:

- official inference is written around the Transformers/PyTorch runtime and
  custom processor/model code
- the existing Qwen driver is shaped around local Rust/Burn code paths

Impact:

- a VibeVoice driver may require a different backend strategy than Qwen
- backend assumptions should not be hidden in shared framework contracts

Mitigation:

- isolate backend decisions under `tts_vibevoice/backend/`
- keep the public driver surface backend-agnostic where possible

## Open Questions to Recheck Before Coding

- what is the smallest complete prompt-object contract that `tts_vibevoice`
  must support
- which parts of the upstream runtime can be hosted natively versus bridged
- which VibeVoice tuning knobs genuinely belong in the public run options
- whether the first verification path should be final-audio only or include a
  private streaming test harness

## Recommended Verification Order

1. Validate model package layout.
2. Validate cached prompt structure.
3. Validate load-time resident runtime creation.
4. Validate a minimal single synthesis request.
5. Validate multi-window progression and stop behavior.
6. Only then consider shared abstraction extraction.

## References

- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- `docs/architecture.md`
- <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
- <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming_inference.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/processor/vibevoice_streaming_processor.py>
