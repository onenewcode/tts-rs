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
| Public facade | `tts_qwen` | `Qwen3TtsPipeline::load()`, `synthesize()`, `synthesize_to_wav()` |
| Text tokenizer / prompt / prefill | `frontend` | `Qwen3TtsPipeline::build_frontend()` |
| Codec generation | `talker` | `generate_talker_tokens()`, `generate_code_predictor_groups()` |
| Waveform decoding | `audio_codec` | `decode_codec_tokens()` |
| Output | `shared::io` | `save_wav()`, `write_wav()` |

## Module Rules

`frontend`, `talker`, and `audio_codec` remain domain modules and depend only on
`shared`. The public crate surface now centers on `Qwen3TtsPipeline`, and the
standalone `tts_cli` crate is a thin wrapper around that facade.

Low-level model structs, remappers, and orchestration helpers stay internal to
the crate unless the facade needs to surface them as part of a high-level
contract.

## Source Layout

```text
tts_qwen/src/
  pipeline.rs       high-level facade and end-to-end orchestration
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
    cli.rs          command args, logging, and facade invocation
    main.rs         thin binary entrypoint
```

## Test Layout

Integration tests live in `tests/` by domain:

- `frontend.rs`
- `tokenizer.rs`
- `pipeline.rs`

Default tests are Rust-only. The real model end-to-end WAV smoke is ignored by
default and can be run explicitly with `cargo test --release -p tts_qwen --test pipeline -- --ignored --nocapture`.
