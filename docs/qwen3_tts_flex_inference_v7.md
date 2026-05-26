# Qwen3-TTS Speech Tokenizer Decoder Inference V7

## Summary

This document defines the audio codec decoder inference: converting quantized codec
token IDs to raw audio waveform via theResidual Vector Quantization (RVQ) decoder.

Status: implemented (forward methods complete, full decoder pipeline functional).

V7 goals:

- implement `forward()` on all decoder model structs (15 total)
- support both single-step `[batch, num_quantizers]` and multi-step `[batch, num_quantizers, time]` input
- produce audio waveform `[batch, 1, num_samples]`
- keep all math inside Burn module/tensor/backend APIs

V7 non-goals:

- encoder inference (audio → tokens, not needed for TTS synthesis)
- streaming/chunked decoding
- custom Conv1d padding strategies (uses Burn's standard Conv1d)
- weight-space changes to the decoder checkpoints

## Architecture

### Data Flow

```
codec_ids [batch, num_quantizers, time_steps]
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Quantizer (RVQ lookup)                          │
│   rvq_first (1 semantic layer)                  │
│   rvq_rest  (15 acoustic layers)                │
│   Per-layer: codebook lookup → sum residuals    │
│   Output: [batch, codebook_dim, time_steps]     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ pre_conv: CausalConv1d(codebook_dim→latent_dim) │
│   Output: [batch, latent_dim, time_steps]       │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Upsample stages (2×)                            │
│   For each ratio in upsampling_ratios:          │
│     CausalTransConv1d(stride=ratio)             │
│     ConvNeXtBlock (dwconv→LN→pw→SiLU→pw→γ+res) │
│   Output: [batch, latent_dim, time*prod(ratios)]│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Pre-Transformer (8-layer decoder)               │
│   input_proj → N×[RMSNorm→Attn+Scale→RMSNorm→   │
│                  SwiGLU+Scale] → RMSNorm→output  │
│   Attention: MHA with RoPE (θ=10000)            │
│   MLP: SwiGLU (silu(gate) * up → down)          │
│   Output: [batch, time, latent_dim]             │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Wave Decoder                                    │
│   InputConv (latent→decoder_dim, k=7)           │
│   4× UpsampleStage:                             │
│     SnakeBeta→TransConv(rate=8,5,4,3)→3×ResUnit │
│   OutputActivation: SnakeBeta                   │
│   OutputConv (96→1, k=7)                        │
│   Output: [batch, 1, total_samples]             │
└─────────────────────────────────────────────────┘
```

### Key Dimensions

| Parameter | Value |
|---|---|
| codebook_dim | 512 |
| latent_dim | 1024 |
| decoder_dim | 1536 |
| hidden_size (transformer) | 512 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 |
| head_dim | 64 |
| intermediate_size (MLP) | 1024 |
| num_hidden_layers | 8 |
| num_quantizers | 16 |
| num_semantic_quantizers | 1 |
| upsample_rates | [8, 5, 4, 3] |
| upsampling_ratios | [2, 2] |
| Total upsample factor | 8×5×4×3 = 480 |

### Forward Methods Implemented

| Struct | Method | File |
|---|---|---|
| `TokenizerSnakeBeta` | `forward(x) → x + sin²(αx)/(β+ε)` | `common.rs` |
| `TokenizerCausalConv1d` | `forward(x) → conv(x)` | `common.rs` |
| `TokenizerCausalTransConv1d` | `forward(x) → trans_conv(x)` | `common.rs` |
| `TokenizerLayerScale` | `forward(x) → x * scale` | `common.rs` |
| `WaveDecoderResidualUnit` | `forward(x) → x + conv2(act2(conv1(act1(x))))` | `wave_decoder.rs` |
| `WaveDecoderUpsampleStage` | `forward(x) → 3×ResUnit(TransConv(Snake(x)))` | `wave_decoder.rs` |
| `WaveDecoderEntry` | `forward(x) → match variant` | `wave_decoder.rs` |
| `ConvNeXtBlock` | `forward(x) → x + γ·pw2(SiLU(pw1(LN(dwconv(x)))))` | `decoder.rs` |
| `DecoderMlp` | `forward(x) → down(silu(gate) * up)` | `decoder.rs` |
| `DecoderAttention` | `forward(x) → MHA with RoPE` | `decoder.rs` |
| `DecoderTransformerLayer` | `forward(x) → x + Scale·Attn(Norm(x)) + Scale·MLP(Norm(x))` | `decoder.rs` |
| `DecoderTransformer` | `forward(x) → output_proj(norm(layers(input_proj(x))))` | `decoder.rs` |
| `DecoderCodebook` | `forward(ids) → gather(embedding_sum/cluster_usage, ids)` | `decoder.rs` |
| `DecoderResidualVectorQuantizer` | `forward(ids) → output_proj(VQ(input_proj(...)))` | `decoder.rs` |
| `DecoderQuantizer` | `forward(ids) → rvq_first + rvq_rest` | `decoder.rs` |
| `Decoder` | `forward(ids) → full pipeline to waveform` | `decoder.rs` |

### Public API

```rust
/// Multi-time-step: [batch, num_quantizers, time_steps] → [batch, 1, samples]
pub fn decode_codec_tokens<B>(loaded, codec_ids, config) -> Result<Tensor<B, 3>>

/// Single-step convenience: [batch, num_quantizers] → [batch, 1, samples]
pub fn decode_codec_tokens_single_step<B>(loaded, codec_ids, config) -> Result<Tensor<B, 3>>
```

## Python Reference Alignment

The audio codec decoder should produce bit-identical output to Python when given
the same codec token IDs. Test approach:

1. Generate codec tokens from Python talker generation
2. Pass same tokens through Python audio codec decoder
3. Pass tokens through Rust `decode_codec_tokens`
4. Compare waveform samples element-wise

Required Python reference script: `py/generate_reference_v7.py`
- Loads audio codec model with `dtype="auto"`
- Takes codec token IDs from V3/V4 reference generation
- Runs decoder forward
- Exports waveform samples (first-100, last-100, max_abs_diff stats)

### Reference Artifacts

- `reference_v7_decoder.json`: codec_ids, expected waveform stats (shape, first_100, last_100, max_abs)

## Test Plan

Required tests:

- (future) `decode_codec_tokens` produces expected waveform shape
- (future) `decode_codec_tokens` waveform matches Python within BF16 tolerance
- (future) `decode_codec_tokens_single_step` result matches `decode_codec_tokens` with `unsqueeze(2)`
- `decode_codec_tokens` rejects wrong num_quantizers count
- (future) multi-time-step: waveform length scales with time_steps

## Acceptance Criteria

- `cargo test -p tts_rs_qwen_burn` passes (41 tests) ✓
- All forward methods compile and execute without panic ✓
- Decoder pipeline runs end-to-end in `e2e_inference` binary ✓
- (future) Waveform output matches Python within BF16 numerical tolerance
- (future) Weight roundtrip test passes (load + export = source)
