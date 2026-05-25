# Qwen3-TTS Production Pipeline V9

## Summary

This document defines the ninth milestone: bridging the remaining gaps between the
current end-to-end pipeline (which produces white noise from zero embeddings) and
a production-ready system that generates real speech from text.

Status: design-finalized, implementation-pending.

Current state (V8):
- Talker generation from placeholder embeddings → white noise audio
- Speech tokenizer decoder loads and runs correctly
- WAV output working (24kHz, 16-bit mono)
- 67 fast tests pass, weight roundtrip verified (402 + 496 tensors)

V9 goals:

- add `shared/io/output.rs` — reusable WAV writer extracted from e2e binary
- add text-side preprocessing in Rust (or Python→Rust embedding bridge)
- fix code predictor dtype mismatch for real codec group expansion
- pass real talker hidden states to code predictor
- validate full pipeline against Python reference for a short text prompt
- produce a recognizable audio output (not white noise)

V9 non-goals:

- speaker encoder / voice cloning
- streaming / real-time inference
- batch inference
- audio post-processing (normalization, denoising)
- multi-format audio output (FLAC, MP3)

## Sub-Stages

### V9a: Audio Output Module

Extract WAV writer from `e2e_inference.rs` into `shared/io/output.rs`.

```rust
/// Write a Burn tensor as a 16-bit PCM WAV file.
pub fn save_wav<B: Backend>(
    waveform: &Tensor<B, 3>,    // [batch, channels, samples]
    path: impl AsRef<Path>,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>>
```

Rust interface:
- `save_wav()` — full WAV file writer with RIFF header
- `write_wav()` — streaming writer for large files

Python baseline:
- `py/generate_reference_v9a.py` — generates known sine wave, exports raw PCM values
- `tests/audio_alignment.rs` — compares Rust WAV output byte-for-byte with Python

### V9b: Text-Side Preprocessing Bridge

Two options for getting real text embeddings into Rust:

**Option A: Python export** — Python tokenizer encodes text, saves embeddings as `.npy` or JSON.
Rust binary loads the file and uses it as input.

**Option B: Rust tokenizer** — Implement tokenizer + embedding lookup in Rust using Burn's
Embedding module. Requires porting the Qwen3 tokenizer config (`tokenizer_config.json`,
`vocab.json`, `merges.txt`).

Recommendation: Start with Option A (lower risk), plan Option B for V10.

Python script: `py/export_text_embeddings.py`
- Takes text prompt + model dir
- Tokenizes text, looks up embeddings
- Exports `text_embeddings.json` with shape + flattened values

### V9c: Code Predictor Dtype Fix

The code predictor fails with "matmul: dtype mismatch left: BF16 right: F32" when
called from the e2e binary. The root cause is in the `small_to_mtp_projection` linear
layer — its weights are F32 while the talker hidden state is BF16.

Fix approach:
1. Audit `Qwen3TtsTalkerCodePredictor` model loading — check dtype of projection weights
2. Add explicit dtype cast in `forward_code_predictor_hidden` before projection
3. Or ensure hidden state is cast to match the projection weight dtype

### V9d: Real Hidden State Passing

Currently the e2e binary passes zero tensors as hidden state to the code predictor.
The correct approach: extract the last hidden state from each talker decode step
and pass it to the code predictor.

Changes to `generate_talker_tokens`:
- Add option to collect hidden states per step
- Return `step_hidden_states: Vec<Tensor<B, 2>>` alongside tokens

Or: use `collect_step_diagnostics = true` which already collects activations, and
extract `"model.norm.output"` as the hidden state.

### V9e: Full Pipeline Validation

End-to-end test with a real text prompt:

1. Python: encode text → embeddings → run full pipeline → reference audio
2. Rust: load embeddings → generate → decode → output audio
3. Compare:
   - Talker token IDs (should match Python greedy generation)
   - Waveform samples (BF16 tolerance)

## Python Reference Data

| Script | Output | Covers |
|---|---|---|
| `py/generate_reference_v9a.py` | `reference_v9a_audio.json` | WAV encoding |
| `py/export_text_embeddings.py` | `text_embeddings.json` | Text → embeddings bridge |
| `py/generate_reference_v9e.py` | `reference_v9e_e2e.json` | Full pipeline with text |

## Test Plan

Required fast tests:

- `save_wav` produces valid RIFF WAV header
- `save_wav` roundtrips: write → read → same samples
- `save_wav` rejects empty waveform
- `save_wav` handles sample rate validation

Required alignment tests:

- `tests/audio_alignment.rs` — WAV bytes match Python (ignored, needs Python baseline)
- `tests/talker_alignment.rs` (V1-V4) — continues to pass with hidden state collection enabled
- `tests/decoder_alignment.rs` (V7) — decoder alignment updated for multi-step input

## Acceptance Criteria

- `cargo test -p tts_rs_qwen_burn` — all fast tests pass
- `save_wav` produces byte-identical output to Python's `wave` module
- Text embeddings bridge works (Python export → Rust load → same values)
- Full pipeline with real text embeddings produces recognizable speech (not white noise)
- (stretch) Talker tokens match Python greedy generation for same text input
