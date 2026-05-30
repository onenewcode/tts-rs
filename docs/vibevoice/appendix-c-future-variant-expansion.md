# Appendix C: Future Variant Expansion

## Purpose

This appendix records how the current `Realtime-0.5B` plan should leave room for
future VibeVoice family work without forcing speculative abstraction into the
first implementation.

## Current Reality

The first concrete target available in this repository is
`VibeVoice-Realtime-0.5B`.

That target has a published profile centered on:

- single-speaker output
- cached-prompt conditioning
- streaming text input
- realtime chunked generation

Future VibeVoice variants may differ along several axes, but those differences
should be treated as extension points, not first-pass implementation goals.

## Extension Axes to Reserve

### 1. Prompt asset kind

Current assumption:

- cached realtime prompt object

Future possibilities:

- richer family-specific prompt bundles
- multi-speaker prompt packs
- other precomputed conditioning artifacts

### 2. Session mode

Current assumption:

- realtime streaming-oriented session behavior

Future possibilities:

- long-form non-realtime generation
- multi-speaker dialogue generation
- different chunking or buffering behavior

### 3. Speaker model

Current assumption:

- one active speaker per request

Future possibilities:

- multiple named speakers
- turn-taking metadata
- conversation-style request surfaces

### 4. Capability profile

Current assumption:

- single-speaker
- cached-prompt based
- English-first release constraints

Future possibilities:

- broader multilingual claims
- multi-speaker routing
- different conditioning methods

### 5. Backend strategy

Current assumption:

- one initial backend path chosen for the first driver landing

Future possibilities:

- alternate local runtimes
- backend-specific acceleration paths
- different hosting strategies for prompt or scheduler state

## What Should Stay Stable Across Variants

Even as the family grows, these high-level choices should stay stable:

- `tts_vibevoice` remains the VibeVoice-family landing crate
- `tts_core` remains the framework host for lifecycle and capabilities
- request semantics stay driver-family specific unless a shared abstraction is
  actually proven
- future extraction into shared code happens after overlap is observed, not
  before

## What Should Not Be Frozen Too Early

The first implementation should not hard-code these assumptions as eternal truth:

- that all VibeVoice variants use the same prompt asset shape
- that all VibeVoice variants are single-speaker
- that all VibeVoice variants should expose the same public request fields
- that realtime diffusion stepping parameters map cleanly to every family member

## Recommended Naming Discipline

Use `tts_vibevoice` as the crate name for the family landing zone, but keep the
first runtime and docs explicit about the concrete target:

- family: VibeVoice
- first landed variant: `VibeVoice-Realtime-0.5B`

That naming keeps future extension natural without pretending the family is
already unified internally.

## References

- `docs/vibevoice/01-scope-and-driver-decision.md`
- `docs/vibevoice/02-model-architecture-and-runtime.md`
- `docs/vibevoice/03-integration-stages.md`
- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- <https://github.com/microsoft/VibeVoice>
- <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
