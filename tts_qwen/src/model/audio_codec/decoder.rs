//! # Audio Codec Decoder
//!
//! Converts quantized codec token IDs to audio waveform through:
//!
//! ```text
//! codec_ids [B, num_quantizers, T]
//!   → Quantizer (RVQ codebook lookup, 16 layers)
//!   → pre_conv (CausalConv1d)
//!   → Decoder Transformer (8-layer, RoPE, SwiGLU)
//!   → Upsample stages (CausalTransConv + ConvNeXt, ×4 time)
//!   → Wave Decoder (4× UpsampleStage + SnakeBeta + output conv)
//!   → waveform [B, 1, total_samples]
//! ```
//!
//! Total time expansion: upsampling_ratios × upsample_rates = 2×2 × 8×5×4×3 = 1920×.
//! For 12.5 Hz codec input this produces 24 kHz audio.

use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear, RmsNorm, RotaryEncoding};
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use super::encoder::Qwen3TtsAudioCodecEncoder;
use super::wave_decoder::Qwen3TtsAudioCodecWaveDecoderEntry;
use crate::kernels::activation::AudioCodecLayerScale;
use crate::kernels::conv::AudioCodecCausalConv1d;

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecCheckpoint<B: Backend> {
    pub decoder: Qwen3TtsAudioCodecDecoder<B>,
    pub encoder: Qwen3TtsAudioCodecEncoder<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoder<B: Backend> {
    pub pre_transformer: Qwen3TtsAudioCodecDecoderTransformer<B>,
    pub quantizer: Qwen3TtsAudioCodecDecoderQuantizer<B>,
    pub pre_conv: AudioCodecCausalConv1d<B>,
    pub upsample: Vec<(
        crate::kernels::conv::AudioCodecCausalTransConv1d<B>,
        Qwen3TtsAudioCodecConvNeXtBlock<B>,
    )>,
    pub decoder: Vec<Qwen3TtsAudioCodecWaveDecoderEntry<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecConvNeXtBlock<B: Backend> {
    pub dwconv: AudioCodecCausalConv1d<B>,
    pub norm: LayerNorm<B>,
    pub pwconv1: Linear<B>,
    pub pwconv2: Linear<B>,
    pub gamma: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecDecoderTransformerLayer<B>>,
    pub norm: RmsNorm<B>,
    pub input_proj: Linear<B>,
    pub output_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsAudioCodecDecoderAttention<B>,
    pub mlp: Qwen3TtsAudioCodecDecoderMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub self_attn_layer_scale: AudioCodecLayerScale<B>,
    pub mlp_layer_scale: AudioCodecLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderQuantizer<B: Backend> {
    pub rvq_first: Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B>,
    pub rvq_rest: Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub vq: Qwen3TtsAudioCodecDecoderResidualVectorQuantization<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderResidualVectorQuantization<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecDecoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderVectorQuantization<B: Backend> {
    pub _codebook: Qwen3TtsAudioCodecDecoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoderCodebook<B: Backend> {
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embedding_sum: Param<Tensor<B, 2>>,
}

// ---- Forward implementations -------------------------------------------------

impl<B: Backend> Qwen3TtsAudioCodecConvNeXtBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let [batch, channels, time] = x.dims();
        // Depthwise conv7 (3D in → 3D out)
        let h = self.dwconv.forward(x);
        // LayerNorm over channel dimension: [B, C, T] → [B*T, C] → LN → [B, C, T]
        let h = h.swap_dims(1, 2).reshape([batch * time, channels]);
        let h = self.norm.forward(h);
        let h = h.reshape([batch, time, channels]).swap_dims(1, 2);
        // Pointwise expand: [B, C, T] → [B*T, C] → Linear → [B*T, out] → [B, out, T]
        let h = h.swap_dims(1, 2).reshape([batch * time, channels]);
        let h = self.pwconv1.forward(h); // [B*T, expand]
        let h = gelu(h);
        let [_, _expand] = h.dims();
        let h = self.pwconv2.forward(h); // [B*T, channels]
        let h = h.reshape([batch, time, channels]).swap_dims(1, 2); // [B, C, T]
        // Gamma scaling (per-channel)
        let gamma = self.gamma.val().reshape([1, channels, 1]);
        let h = h.mul(gamma);
        residual + h
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderMlp<B> {
    /// SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();
        let x2d = x.reshape([batch * seq_len, hidden]);
        let gate = self.gate_proj.forward(x2d.clone());
        let up = self.up_proj.forward(x2d);
        let activated = silu(gate);
        let product = activated * up;
        let down = self.down_proj.forward(product);
        down.reshape([batch, seq_len, hidden])
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderAttention<B> {
    /// Multi-head self-attention with RoPE.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = x.dims();
        let x2d = x.reshape([batch_size * seq_len, hidden_size]);

        let q = self
            .q_proj
            .forward(x2d.clone())
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .swap_dims(1, 2); // [batch, heads, seq, dim]
        let k = self
            .k_proj
            .forward(x2d.clone())
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x2d)
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);

        // Apply RoPE
        let offset = 0; // No KV cache for audio codec decoder (full-sequence)
        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        // Repeat KV heads to match Q heads
        let n_rep = num_heads / num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        // Scaled dot-product attention
        let dtype = q.dtype();
        let attn_scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let attn_scores = if let Some(mask) = mask {
            attn_scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attn_scores
        };
        let attn_weights = softmax(attn_scores.cast(DType::F32), 3).cast(dtype);
        let attn_output = attn_weights.matmul(v);

        // Back to [batch, seq, hidden]
        let output = attn_output
            .swap_dims(1, 2) // [batch, seq, heads, dim]
            .clone()
            .reshape([batch_size, seq_len, num_heads * head_dim]);

        let out2d = output.reshape([batch_size * seq_len, num_heads * head_dim]);
        self.o_proj
            .forward(out2d)
            .reshape([batch_size, seq_len, hidden_size])
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderTransformerLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();

        // Self-attention with pre-norm + layer scale
        let normed = rms_norm_3d(&self.input_layernorm, x);
        let attn_out =
            self.self_attn
                .forward(normed, num_heads, num_kv_heads, head_dim, rope, mask);
        let attn_out = self.self_attn_layer_scale.forward(attn_out);
        let x = residual + attn_out;

        // MLP with pre-norm + layer scale
        let residual = x.clone();
        let normed = rms_norm_3d(&self.post_attention_layernorm, x);
        let mlp_out = self.mlp.forward(normed);
        let mlp_out = self.mlp_layer_scale.forward(mlp_out);
        residual + mlp_out
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderTransformer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq, latent] = x.dims();

        // Project input to transformer hidden size
        let x2d = x.reshape([batch * seq, latent]);
        let proj_in = self.input_proj.forward(x2d); // [batch*seq, hidden]
        let [_, hidden_size] = proj_in.dims();
        let mut h = proj_in.reshape([batch, seq, hidden_size]); // [batch, seq, hidden]

        for layer in &self.layers {
            h = layer.forward(h, num_heads, num_kv_heads, head_dim, rope, mask.clone());
        }

        // Final RMS norm
        let h_normed = rms_norm_3d(&self.norm, h);
        let [b, s, hs] = h_normed.dims();
        let h2d = h_normed.reshape([b * s, hs]);
        let out = self.output_proj.forward(h2d); // [b*s, latent]
        let [_, out_dim] = out.dims();
        out.reshape([b, s, out_dim])
    }
}

// -- Quantizer (codebook lookup) -----------------------------------------------

impl<B: Backend> Qwen3TtsAudioCodecDecoderCodebook<B> {
    /// Look up embedding vectors for given token indices.
    /// `token_ids`: [batch_size, seq_len] integer tensor
    /// Returns: [batch_size, embed_dim, seq_len]
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = token_ids.dims();
        let [_codebook_size, embed_dim] = self.embedding_sum.dims();
        // Normalize embeddings: embedding_sum / cluster_usage
        let usage = self.cluster_usage.val().unsqueeze_dim(1); // [cb_size, 1]
        let codebook = self.embedding_sum.val().div(usage); // [cb_size, embed_dim]
        // Gather: codebook[token_ids] → [batch, seq, embed_dim]
        let flat_ids = token_ids.reshape([batch * seq]);
        let gathered = codebook.select(0, flat_ids); // [batch*seq, embed_dim]
        let result: Tensor<B, 3> = gathered.reshape([batch, seq, embed_dim]);
        result.swap_dims(1, 2) // [batch, channels, time]
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderVectorQuantization<B> {
    /// Look up codebook embedding and return the vector.
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self._codebook.forward(token_ids)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderResidualVectorQuantization<B> {
    /// Sum of codebook embeddings across all layers.
    pub fn forward(&self, token_ids: &[Tensor<B, 2, Int>]) -> Tensor<B, 3> {
        let mut out: Option<Tensor<B, 3>> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let emb = layer.forward(token_ids[layer_idx].clone());
            out = Some(match out {
                Some(acc) => acc + emb,
                None => emb,
            });
        }
        out.expect("Residual VQ must have at least one layer")
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B> {
    /// Decode token IDs to embedding vectors.
    ///
    /// For decoder inference, we skip `input_proj` (which compresses encoder output
    /// from codebook_dim→hidden before quantization). Codebook lookup already produces
    /// `hidden`-dim vectors, so we go straight to `output_proj` (hidden→codebook_dim).
    pub fn forward_decode(&self, token_ids: &[Tensor<B, 2, Int>]) -> Tensor<B, 3> {
        let vq_out = self.vq.forward(token_ids); // [batch, hidden, time]
        self.output_proj.forward(vq_out) // Conv1d: hidden -> codebook_dim
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderQuantizer<B> {
    /// Decode codec tokens to continuous embedding.
    /// `codec_ids`: [batch, num_quantizers, time_steps] (3D) — one token per quantizer
    ///   layer for each time step.
    /// Returns: [batch, codebook_dim, time_steps]
    pub fn forward(
        &self,
        codec_ids: Tensor<B, 3, Int>,
        num_semantic_quantizers: usize,
    ) -> Tensor<B, 3> {
        let [batch, _num_quantizers, time_steps] = codec_ids.dims();
        let total_layers = self.rvq_first.vq.layers.len() + self.rvq_rest.vq.layers.len();
        // Slice per layer: [batch, 1, time_steps] → reshape to [batch, time_steps]
        let per_layer_tokens: Vec<Tensor<B, 2, Int>> = (0..total_layers)
            .map(|i| {
                codec_ids
                    .clone()
                    .slice([0..batch, i..i + 1, 0..time_steps])
                    .reshape([batch, time_steps])
            })
            .collect();

        let semantic_tokens: &[Tensor<B, 2, Int>] = &per_layer_tokens[..num_semantic_quantizers];
        let acoustic_tokens: &[Tensor<B, 2, Int>] = &per_layer_tokens[num_semantic_quantizers..];

        let sem = self.rvq_first.forward_decode(semantic_tokens);
        let aco = self.rvq_rest.forward_decode(acoustic_tokens);
        sem + aco
    }
}

// -- Top-level decoder ---------------------------------------------------------

impl<B: Backend> Qwen3TtsAudioCodecDecoder<B> {
    /// Full decoder forward: codec tokens → audio waveform.
    ///
    /// `codec_ids`: [batch, num_quantizers, time_steps] — one token per quantizer layer
    ///   for each time step. For single-step, use shape `[batch, num_quantizers, 1]`.
    pub fn forward(
        &self,
        codec_ids: Tensor<B, 3, Int>,
        num_semantic_quantizers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        // 1. Quantizer: codec IDs → embedding [batch, codebook_dim, time_steps]
        let q_out = self.quantizer.forward(codec_ids, num_semantic_quantizers);

        // 2. pre_conv: causal conv along time dimension
        let h = self.pre_conv.forward(q_out);

        // 3. Pre-transformer: process time sequence
        // Swap [batch, channels, time] → [batch, time, channels]
        let h_tr: Tensor<B, 3> = h.swap_dims(1, 2).clone();
        let mut h =
            self.pre_transformer
                .forward(h_tr, num_heads, num_kv_heads, head_dim, rope, None);
        // Back to [batch, channels, time]
        h = h.swap_dims(1, 2).clone();

        // 4. Upsample stages: CausalTransConv + ConvNeXt blocks
        for (trans_conv, convnext) in &self.upsample {
            h = trans_conv.forward(h);
            h = convnext.forward(h);
        }

        // 5. Wave decoder: sequence of conv/upsample entries
        for entry in &self.decoder {
            h = entry.forward(h);
        }
        h.clamp_min(-1.0).clamp_max(1.0)
    }

    /// Convenience: single-time-step decoder forward.
    /// `codec_ids`: [batch, num_quantizers] — reshaped to [batch, num_quantizers, 1].
    pub fn forward_single_step(
        &self,
        codec_ids: Tensor<B, 2, Int>,
        num_semantic_quantizers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let [batch, num_q] = codec_ids.dims();
        let codec_3d = codec_ids.reshape([batch, num_q, 1]);
        self.forward(
            codec_3d,
            num_semantic_quantizers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        )
    }
}

// -- Helpers -------------------------------------------------------------------

/// Repeat key/value heads to match query head count (grouped-query attention).
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
        batch_size,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    ])
}

/// RMS normalization on 3D tensor [batch, seq, channels].
fn rms_norm_3d<B: Backend>(norm: &RmsNorm<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, channels] = x.dims();
    let x2d = x.reshape([batch * seq, channels]);
    let out2d = norm.forward(x2d);
    out2d.reshape([batch, seq, channels])
}
