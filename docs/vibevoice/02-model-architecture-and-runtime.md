# Model Architecture and Runtime

## Purpose

This document captures what is currently known about the architecture of
`VibeVoice-Realtime-0.5B`, separating direct observations from implementation
inference.

The goal is not to restate upstream marketing copy. The goal is to define the
runtime reality that `tts_vibevoice` will need to host.

## Direct Observations

### Upstream model positioning

`dir/microsoft/VibeVoice-Realtime-0.5B/README.md` and the upstream realtime doc
both describe the model as:

- realtime TTS
- streaming text input
- single-speaker output
- long-form generation relative to other realtime systems
- diffusion-based acoustic latent generation
- an interleaved, windowed design

The same documents explicitly say the realtime model removes the semantic
Tokenizer and relies on an acoustic tokenizer running at an ultra-low frame
rate of 7.5 Hz.

### Local configuration data

`dir/microsoft/VibeVoice-Realtime-0.5B/config.json` shows:

- `model_type: vibevoice_streaming`
- architecture name `VibeVoiceStreamingForConditionalGenerationInference`
- `decoder_config.hidden_size = 896`
- `decoder_config.num_hidden_layers = 24`
- `decoder_config.max_position_embeddings = 8192`
- `diffusion_head_config.head_layers = 4`
- `diffusion_head_config.latent_size = 64`
- `acoustic_vae_dim = 64`
- `tts_backbone_num_hidden_layers = 20`

`dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json` shows:

- `processor_class = VibeVoiceStreamingProcessor`
- `speech_tok_compress_ratio = 3200`
- `audio_processor.sampling_rate = 24000`
- `language_model_pretrained_name = Qwen/Qwen2.5-0.5B`

### Local weight layout observations

Inspection of `dir/microsoft/VibeVoice-Realtime-0.5B/model.safetensors` shows
these prominent module families:

- `model.language_model.*`
- `model.tts_language_model.*`
- `model.prediction_head.*`
- `model.acoustic_connector.*`
- `model.acoustic_tokenizer.decoder.*`
- `tts_eos_classifier.*`

It does not show a corresponding
`model.acoustic_tokenizer.encoder.*` family in the shipped release.

### Official demo and processor behavior

The upstream realtime demo and processor show that inference is driven by:

- loading a cached speaker prompt from a `.pt` asset
- calling `process_input_with_cached_prompt(...)`
- passing `all_prefilled_outputs` into `model.generate(...)`

The official inference implementation defines:

- `TTS_TEXT_WINDOW_SIZE = 5`
- `TTS_SPEECH_WINDOW_SIZE = 6`

## Architectural Inference

The following points are inferences grounded in the observations above.

### The runtime is not Qwen-style codec-token generation

The current Qwen driver in this repo works as:

- compile request text and control tokens
- run an autoregressive talker
- produce discrete codec token IDs
- decode those IDs into waveform audio

VibeVoice-Realtime does not fit that pattern.

The upstream implementation instead uses:

- a text path through a language-model stage
- a TTS-specific hidden-state stage
- a diffusion head that predicts continuous acoustic latents
- an acoustic decoder that turns those latents into audio chunks

This means `tts_vibevoice` should not inherit the internal mental model of
`tts_qwen3_tts`.

### The model appears split into a text stack and a TTS stack

The local weights contain:

- 4 layers under `model.language_model.layers.*`
- 20 layers under `model.tts_language_model.layers.*`

Together with `tts_backbone_num_hidden_layers = 20` and a total decoder depth
of 24, the strongest reading is:

- an early text-oriented language-model stage
- a later TTS-oriented backbone stage

This is consistent with the upstream description of incrementally encoding text
while continuing speech-latent generation from prior context.

### The prompt is a cached runtime artifact, not a plain reference audio input

The official realtime demo does not feed a user audio file directly into the
model at inference time. It loads a cached `.pt` voice preset and passes that
object through both processor and generation paths.

That means the visible public runtime contract is closer to:

- text input
- cached prompt asset
- generation parameters

and not to:

- text input
- arbitrary reference audio file
- on-the-fly voice cloning

### The shipped release is decode-centric for speech conditioning

Because the local weight file includes `model.acoustic_tokenizer.decoder.*` but
not the encoder family, and because the model card says the release removes the
acoustic tokenizer needed for users to create their own embeddings, the shipped
release should be treated as decode-centric.

In practical framework terms, that means first-class support should target the
published cached-prompt path before any design assumes local arbitrary voice
prompt construction.

## Resulting Runtime Model

A simplified `tts_vibevoice` mental model should be:

```text
cached prompt + input text
    -> text tokenization and prompt prefill
    -> language-model stage
    -> TTS backbone stage
    -> diffusion head
    -> continuous acoustic latents
    -> acoustic decoder
    -> audio chunks / final waveform
```

This is intentionally different from the Qwen mental model:

```text
text
    -> request compiler
    -> autoregressive talker
    -> discrete codec token ids
    -> codec decoder
    -> final waveform
```

## Implications for `tts_vibevoice`

The driver design should assume:

- request-time cached prompt assets are part of the normal flow
- streaming session logic is core, not optional implementation detail
- generated intermediate state is continuous latent-driven rather than discrete
  codec-token driven
- capability projection should say single-speaker and cached-prompt oriented,
  not generic voice cloning by default
- the first release should model realtime behavior first, then generalize only
  when more VibeVoice variants are actually in hand

## References

Local:

- `dir/microsoft/VibeVoice-Realtime-0.5B/README.md`
- `dir/microsoft/VibeVoice-Realtime-0.5B/config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/preprocessor_config.json`
- `dir/microsoft/VibeVoice-Realtime-0.5B/model.safetensors`
- `tts_qwen3_tts/src/model/mod.rs`

Official:

- <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md>
- <https://github.com/microsoft/VibeVoice/blob/main/demo/realtime_model_inference_from_file.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming_inference.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modeling_vibevoice_streaming.py>
- <https://github.com/microsoft/VibeVoice/blob/main/vibevoice/processor/vibevoice_streaming_processor.py>
