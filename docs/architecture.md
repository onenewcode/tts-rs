# TTS Inference Engine Architecture

`tts_qwen` is a Rust inference engine for the local Qwen3-TTS 12Hz
0.6B CustomVoice path. `tts_cli` is the workspace CLI wrapper around that
library.

## Pipeline

```text
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
`shared`. The standalone `tts_cli` crate composes those domains into the
single-sample command-line workflow.

The public API intentionally stays as narrow free functions rather than a session
facade. Old `speech_tokenizer` public paths and migration-only verification
helpers are not re-exported.

## Source Layout

```text
tts_qwen/src/
  frontend/        text tokenizer, CustomVoice prompt, prefill tensors
  talker/          autoregressive talker and code predictor generation
  audio_codec/     codec token to waveform decoder
  shared/
    config/        model config structs
    io/            checkpoint loading and WAV output
    nn/            shared layers and tensor helpers
    runtime/       sampling and KV cache
tts_cli/
  src/
    cli.rs          command args, logging, and end-to-end orchestration
    main.rs         thin binary entrypoint
```

## Test Layout

Integration tests live in `tests/` by domain:

- `frontend.rs`
- `tokenizer.rs`
- `prefill.rs`
- `pipeline.rs`

Default tests are Rust-only. The real model end-to-end WAV smoke is ignored by
default and can be run explicitly with `cargo test --release -p tts_qwen --test pipeline -- --ignored --nocapture`.
