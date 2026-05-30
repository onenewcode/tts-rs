# Appendix A: Cached Prompt and Assets

## Purpose

This appendix isolates the asset model behind `VibeVoice-Realtime-0.5B`, with
special focus on cached prompt handling.

That topic deserves its own appendix because cached prompts are not a small
optional feature in the published realtime flow. They are a primary part of the
normal runtime contract.

## Direct Asset Observations

### Model directory contents

The local model directory currently contains:

- `README.md`
- `config.json`
- `configuration.json`
- `preprocessor_config.json`
- `model.safetensors`
- `figures/`

This differs materially from the current Qwen package layout expected by
`tts_qwen3_tts`, which looks for tokenizer files, generation config files, and a
nested `speech_tokenizer/` directory.

### Processor metadata

`preprocessor_config.json` identifies:

- `VibeVoiceStreamingProcessor`
- `speech_tok_compress_ratio = 3200`
- 24 kHz audio processing
- `Qwen/Qwen2.5-0.5B` as the language-model pretrained source

### Prompt-source behavior in the official realtime flow

The official demo:

- scans a `demo/voices/streaming_model` directory
- chooses a speaker preset
- loads that preset from a `.pt` file
- sends it both to `process_input_with_cached_prompt(...)` and to
  `model.generate(..., all_prefilled_outputs=...)`

This is strong evidence that a cached prompt is a structured runtime object,
not merely a speaker ID.

## Design Interpretation

### Cached prompt belongs to the request path

The model package should describe the runtime itself.
The cached prompt should describe the selected voice conditioning for one run.

That makes cached prompts request-time assets rather than package-level assets.

This separation matters because:

- multiple prompts can be used with the same loaded model
- prompt validation errors should not be reported as package corruption
- prompt storage location may vary by deployment

### Prompt validation should be explicit

The first `tts_vibevoice` implementation should not treat a prompt path as an
opaque blob. It should validate prompt structure as early as possible.

At minimum, the validation strategy should be capable of detecting:

- wrong file type
- unreadable or malformed object data
- missing required prefilled sub-structures
- prompt/runtime mismatch

## Proposed Asset Contract for `tts_vibevoice`

### Model package contract

First-pass required model artifacts:

- `config.json`
- `configuration.json`
- `preprocessor_config.json`
- `model.safetensors`

First-pass optional supporting artifacts:

- documentation files
- local helper assets
- any future local tokenizer mirrors if explicitly supported later

### Request-time prompt contract

First-pass request-time prompt input should support:

- prompt path input
- optionally, a prepared prompt object path inside a higher-level app flow if
  that becomes useful later

The first implementation should document the prompt as a VibeVoice-specific
conditioning asset, not as a generic cross-model voice preset.

## Why This Must Stay Separate From Qwen Concepts

`tts_qwen3_tts` currently exposes voice-clone concepts centered around:

- reference audio
- transcript hints
- speaker embedding creation
- optional reference codec frames

That is a different conditioning model.

For VibeVoice-Realtime, the public published path is cached-prompt based. Until
that changes, the docs and code should not erase the distinction.

## References

Local:

- `tts_qwen3_tts/src/loading/package/normalize.rs`
- `tts_qwen3_tts/src/surface/request/voice_clone.rs`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/model.safetensors`

Official:

- <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/processor/vibevoice_streaming_processor.py>
- <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
