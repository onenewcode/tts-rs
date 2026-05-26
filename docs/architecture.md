# TTS Inference Engine Architecture

`tts_rs_qwen_burn` is a Rust inference engine for the local Qwen3-TTS 12Hz
0.6B CustomVoice path.

## Pipeline

```
text -> frontend -> talker -> code predictor -> audio_codec -> WAV
```

| Stage | Module | Key API |
|---|---|---|
| Text tokenizer / prompt / prefill | `frontend` | `build_custom_voice_prefill_batch()` |
| Codec generation | `talker` | `generate_talker_tokens()`, `generate_code_predictor_groups()` |
| Waveform decoding | `audio_codec` | `load_qwen3_tts_audio_codec()`, `decode_codec_tokens()` |
| Output | `shared::io` | `save_wav()`, `write_wav()` |

## Module Rules

`frontend`, `talker`, and `audio_codec` are domain modules and depend only on
`shared`. `pipeline` or binaries are responsible for composing those domains into
an end-to-end workflow.

The public API intentionally stays as narrow free functions rather than a session
facade. Old `speech_tokenizer` public paths are not re-exported.

## Source Layout

```
tts_rs_qwen_burn/src/
  frontend/        text tokenizer, CustomVoice prompt, prefill tensors
  talker/          autoregressive talker and code predictor generation
  audio_codec/     codec token to waveform decoder
  shared/
    config/        model config structs
    io/            checkpoint loading and WAV output
    nn/            shared layers and tensor helpers
    runtime/       sampling and KV cache
    verify/        checkpoint verification
  bin/
    qwen3-tts.rs
```

## Test Layout

Integration tests live in `tests/` by domain:

- `frontend.rs`
- `alignment_tokenizer.rs`
- `alignment_prefill.rs`
- `alignment_e2e.rs` (`#[ignore]` for full checkpoint comparison)
- legacy talker/audio codec alignment and roundtrip tests while V9 converges
