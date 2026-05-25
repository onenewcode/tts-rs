# Qwen3-TTS End-to-End Inference Pipeline V8

## Summary

This document defines the final Rust inference milestone: connecting all previous stages
into a single end-to-end TTS pipeline that generates audio waveform from text embeddings.

Status: implemented (builds and runs, validation-blocked on real text embeddings).

V8 goals:

- load both models (talker + speech tokenizer) in a single binary
- run talker autoregressive generation (V3 + V5 sampling/stopping)
- expand codec groups via code predictor (V4)
- decode accumulated codec tokens to waveform (V7)
- save output as 24kHz 16-bit mono WAV

V8 non-goals:

- text-side preprocessing (text → token IDs → embeddings, currently Python-side)
- streaming/real-time inference
- batching across multiple utterances
- voice cloning or voice design (speaker encoder)

## Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                      E2E Inference                           │
├──────────────────────────────────────────────────────────────┤
│  1. Load talker model (config + weights)                     │
│  2. Load speech tokenizer model (config + weights)           │
│                                                              │
│  3. [Python-side] Text → token IDs → input embeddings        │
│                                                              │
│  4. Talker prefill                                           │
│     input_embeds [batch, prefill_len, hidden]                │
│     position_ids  [3, batch, prefill_len]                    │
│     → logits [batch, prefill_len, vocab]                     │
│     → select first token                                     │
│                                                              │
│  5. Autoregressive decode loop (max_new_tokens steps):        │
│     a. Embed selected token → codec_embedding                │
│     b. Decode step (attention + KV cache)                    │
│     c. sample_token (greedy or with V5 sampling controls)    │
│     d. [V6] Apply repetition penalty to logits               │
│     e. [V5] Check EOS, break if stopped                      │
│                                                              │
│  6. Code predictor expansion (per time step):                │
│     a. Embed base codec token + talker hidden state          │
│     b. Teacher-forced prefill (2 tokens)                     │
│     c. Autoregressive decode (num_code_groups - 1 steps)     │
│     d. Concatenate: [base_token, predictor_tokens]           │
│     → codec_ids [batch, num_code_groups]                     │
│                                                              │
│  7. Stack all time steps:                                    │
│     → codec_ids [batch, num_quantizers, time_steps]          │
│                                                              │
│  8. Speech tokenizer decoder:                                │
│     Quantizer → pre_conv → Upsample → Transformer →          │
│     Wave Decoder → waveform [batch, 1, samples]              │
│                                                              │
│  9. Save WAV (24kHz, 16-bit mono PCM)                        │
└──────────────────────────────────────────────────────────────┘
```

## Binary

`src/bin/e2e_inference.rs`

```bash
# Run with local model directory
cargo run --bin e2e_inference --release -- \
    ../Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    output/
```

### Current Limitations

1. **Placeholder embeddings**: Uses zero embeddings (not real text). Real text processing
   requires tokenization + embedding lookup (Python-side for now).
2. **Placeholder hidden states**: Code predictor uses zero hidden states instead of
   talker decode hidden states. A future version should pass the actual hidden state
   from each decode step.
3. **No speaker encoder**: Voice cloning/design not supported.

### Key Dimensions

| Step | Input Shape | Output Shape |
|---|---|---|
| Prefill | `[1, 5, 1024]` (BF16) | `[1, 5, 4096]` |
| Generation | 10 steps | `[1, 10]` token IDs |
| Code predictor | `[1, 1]` token × 10 steps | `[1, 4]` codec groups × 10 |
| Stack | 10 × `[1, 4]` tensors | `[1, 4, 10]` |
| Decoder | `[1, 4, 10]` | `[1, 1, ~24000]` samples |

## Python Reference Alignment

Full end-to-end comparison requires:

1. Generate text embeddings in Python (identical input to Rust)
2. Run Python `Qwen3TTSModel.generate_custom_voice()` 
3. Compare:
   - Generated token IDs (talker + code predictor)
   - Waveform samples (speech tokenizer decoder)
   - Audio quality (subjective listening or objective metrics like MCD)

Required Python reference script: `py/generate_reference_v8.py`
- Takes a fixed text prompt and model directory
- Exports: text embeddings, talker tokens, code predictor tokens, waveform samples
- Format: `reference_v8_e2e.json`

## Test Plan

- (future) `e2e_inference` binary runs without panic
- (future) Token IDs match Python greedy generation
- (future) Waveform samples match within BF16 tolerance
- (future) Output WAV is valid (can be played back)

## Acceptance Criteria

- `cargo test -p tts_rs_qwen_burn` passes (41 tests) ✓
- `cargo build --bin e2e_inference` succeeds ✓
- Binary loads both models and runs full pipeline ✓
- (future) Token-level match with Python reference
- (future) Waveform sample match within BF16 tolerance
