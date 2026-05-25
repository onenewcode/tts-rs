# TTS Inference Engine — Architecture

## Overview

`tts_rs_qwen_burn` is a Rust inference engine for Qwen3-TTS models built on the
[Burn](https://burn.dev) deep learning framework. It converts text embeddings to
audio waveform through a configurable multi-stage pipeline.

## Pipeline

```
config.json → load weights → generate codec tokens → decode to waveform → save WAV
```

| Stage | Domain | Key Functions |
|---|---|---|
| Load | `shared::io` | `load_talker()`, `load_tokenizer()` |
| Generate | `talker` | `generate_talker_tokens()`, `generate_code_predictor_groups()` |
| Decode | `tokenizer` | `decode_codec_tokens()` |
| Output | `shared::io` | `save_wav()` |

## Architecture Decisions

These 8 decisions define the crate structure:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Module organization | **Domain modules** | `talker/` and `tokenizer/` align with TTS concepts |
| 2 | Domain internal structure | **Layered** (model/inference/factory) | Separates compute graph from orchestration |
| 3 | Constructor placement | **Factory functions** | Complex multi-step construction isolated from model |
| 4 | nn/ location | **shared/nn/ + domain extensions** | Shared attention/MLP, domain-specific RoPE |
| 5 | shared/ structure | **Grouped by responsibility** | io/, config/, runtime/, verify/ subgroups |
| 6 | Test organization | **Domain tests.rs + integration tests/** | Fast unit tests in-domain, slow alignment tests in tests/ |
| 7 | Public API | **Free functions** | Maximum flexibility for callers |
| 8 | Multi-model support | **Config-driven** | Same code for 0.6B and 1.5B; new model = new directory |

## Target Structure

```
tts_rs_qwen_burn/src/
  lib.rs

  talker/                 — codec token generation
    mod.rs
    model.rs              — TalkerModel, CodePredictor struct + forward
    inference.rs          — generate_talker_tokens, code predictor loops
    factory.rs            — build_talker(), build_code_predictor()
    types.rs              — Input/Output types
    nn/rope.rs            — M-RoPE (talker-specific)

  tokenizer/              — waveform decoding
    mod.rs
    model.rs              — Decoder, Quantizer, Codebook struct + forward
    transformer.rs        — DecoderTransformer, Attention, MLP layers
    wave.rs               — WaveDecoder, ResidualUnit, UpsampleStage
    inference.rs          — decode_codec_tokens()
    factory.rs            — build_decoder(), build_quantizer()
    types.rs

  shared/
    config/               — Config types (TalkerConfig, DecoderConfig)
    nn/                   — Shared NN primitives (attention, layer, mlp, conv, activation)
    io/                   — load (safetensors), output (WAV)
    runtime/              — sampling, KV cache
    verify/               — weight verification, manifest
    paths.rs              — model directory discovery
    error.rs              — error types

  bin/
    e2e.rs                — end-to-end pipeline
    verify_talker.rs
    verify_tokenizer.rs

tests/
  common/mod.rs
  talker_alignment.rs     — V1-V4
  talker_sampling.rs      — V5-V6
  decoder_alignment.rs    — V7
  roundtrip.rs            — weight roundtrip
```

## Dependency Rules

```
shared/          ← zero internal dependencies
talker/          ← depends on shared/ only
tokenizer/       ← depends on shared/ only
bin/             ← depends on talker + tokenizer + shared
```

No circular dependencies. `talker/` and `tokenizer/` never import from each other.

## Key Design Rules

1. **Factory files** contain only construction logic — no forward, no inference, no I/O
2. **Model files** contain struct definitions + `forward()` methods — no weight loading, no config parsing
3. **Inference files** contain orchestration loops — no tensor math beyond sampling
4. **Config-driven dimensions**: all model sizes come from `config.json`
5. **Free functions API**: public functions are standalone, not methods on a session object
6. **Tests by speed**: unit tests in-domain (`#[cfg(test)]`), integration tests in `tests/` (`#[ignore]`)

## Multi-Model Support

All Qwen3-TTS variants (0.6B, 1.5B, 12Hz, 25Hz) share the same code. Dimensions
are read from each model directory's `config.json` at load time. Adding a new variant
requires no code changes — just a new model directory.

Future Qwen versions with different architectures can be added as new constructors in
the existing factory files, or as new modules if the architecture diverges significantly.
