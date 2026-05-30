# Appendix B: Streaming Session and Diffusion

## Purpose

This appendix captures the runtime loop that makes VibeVoice-Realtime different
from the existing Qwen3 driver.

## Upstream Runtime Signals

The official streaming inference implementation exposes several direct clues:

- `TTS_TEXT_WINDOW_SIZE = 5`
- `TTS_SPEECH_WINDOW_SIZE = 6`
- generation accepts `all_prefilled_outputs`
- the model keeps a TTS EOS classifier
- generation uses a diffusion head and a DPM solver scheduler

The local model config reinforces that structure with:

- a 4-layer diffusion head
- 64-dimensional acoustic latent space
- 24-layer decoder stack split by `tts_backbone_num_hidden_layers = 20`

## Reconstructed Runtime Loop

A simplified architecture-aware reading of the upstream runtime is:

```text
cached prompt + input text
    -> prepare inputs and prefilled state
    -> advance a text window through LM and TTS-LM paths
    -> repeatedly sample speech latents for a speech window
    -> decode latents into audio chunks
    -> feed resulting speech state back into the TTS path
    -> repeat until EOS or another stop condition
```

This differs from the Qwen path, which is effectively:

```text
compiled prompt
    -> autoregressive codec token generation
    -> final codec decode
```

## Why a Session Model Is Required

The VibeVoice runtime has explicit progression structure:

- a prompt-prefilled starting point
- text-window advancement
- speech-window advancement
- termination checks
- audio accumulation or chunk emission

That makes it naturally session-oriented.

Even if the first public `tts_vibevoice` API returns a final `PcmAudio`, the
internal implementation should still model these runtime phases explicitly.

## Diffusion-Specific Consequences

Because the diffusion head predicts continuous latents rather than discrete
codec token IDs:

- intermediate speech state is not naturally represented as a codec token list
- generated output state should not be forced into Qwen-style codec-token
  abstractions
- tuning knobs such as CFG scale and DDPM inference steps are first-class
  runtime concerns

## Recommended Internal State Buckets

For the first `tts_vibevoice` design, keep these internal state buckets
separate:

- prompt-prefilled state
- current text-window progress
- current speech-window progress
- diffusion scheduler state
- acoustic decoder cache state
- final accumulated audio or chunk buffer

## Suggested Session Phase Names

Suggested internal phase names:

- `Prefilled`
- `AdvancingTextWindow`
- `GeneratingSpeechWindow`
- `TerminalReached`
- `Finished`

The names can change later, but the phase model should remain explicit.

## References

Local:

- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/model.safetensors`
- `tts_qwen3_tts/src/model/mod.rs`
- `tts_qwen3_tts/src/execution/session.rs`

Official:

- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming_inference.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modular_vibevoice_text_tokenizer.py>
