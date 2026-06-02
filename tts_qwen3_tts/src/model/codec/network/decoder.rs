use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{Linear, RmsNorm, RotaryEncoding};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use super::activation::AudioCodecLayerScale;
use super::codebook::{gather_codebook_embeddings, normalized_codebook_centroids};
use super::conv::AudioCodecCausalConv1d;
use super::wave::{Qwen3TtsAudioCodecConvNeXtBlock, Qwen3TtsAudioCodecWaveDecoderEntry};

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoder<B: Backend> {
    pub pre_transformer: Qwen3TtsAudioCodecDecoderTransformer<B>,
    pub quantizer: Qwen3TtsAudioCodecDecoderQuantizer<B>,
    pub pre_conv: AudioCodecCausalConv1d<B>,
    pub upsample: Vec<(
        super::conv::AudioCodecCausalTransConv1d<B>,
        Qwen3TtsAudioCodecConvNeXtBlock<B>,
    )>,
    pub decoder: Vec<Qwen3TtsAudioCodecWaveDecoderEntry<B>>,
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

impl<B: Backend> Qwen3TtsAudioCodecDecoderMlp<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(hidden.clone());
        let up = self.up_proj.forward(hidden);
        let activated = silu(gate);
        let product = activated * up;
        self.down_proj.forward(product)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderAttention<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<&Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = hidden.dims();

        let query = self
            .q_proj
            .forward(hidden.clone())
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(hidden.clone())
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(hidden)
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);

        let query = rope.apply(query, 0);
        let key = rope.apply(key, 0);
        let repetitions = num_heads / num_kv_heads;
        let key = key
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repetitions)
            .reshape([batch_size, num_kv_heads * repetitions, seq_len, head_dim]);
        let value = value
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repetitions)
            .reshape([batch_size, num_kv_heads * repetitions, seq_len, head_dim]);

        let dtype = query.dtype();
        let attention_scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let attention_scores = if let Some(mask) = mask {
            attention_scores.mask_fill(mask.clone(), f32::NEG_INFINITY)
        } else {
            attention_scores
        };
        let attention_weights = if dtype == DType::F32 {
            softmax(attention_scores, 3)
        } else {
            softmax(attention_scores.cast(DType::F32), 3).cast(dtype)
        };
        let attention_output = attention_weights.matmul(value);
        let output =
            attention_output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);
        self.o_proj.forward(output)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderTransformerLayer<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<&Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.input_layernorm.forward(hidden);
        let hidden = self
            .self_attn
            .forward(hidden, num_heads, num_kv_heads, head_dim, rope, mask);
        let hidden = self.self_attn_layer_scale.forward(hidden);
        let hidden = residual + hidden;

        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(hidden);
        let hidden = self.mlp.forward(hidden);
        let hidden = self.mlp_layer_scale.forward(hidden);
        residual + hidden
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderTransformer<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<&Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let mut hidden = self.input_proj.forward(hidden);

        for layer in &self.layers {
            hidden = layer.forward(hidden, num_heads, num_kv_heads, head_dim, rope, mask);
        }

        let hidden = self.norm.forward(hidden);
        self.output_proj.forward(hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderCodebook<B> {
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let codebook =
            normalized_codebook_centroids(self.cluster_usage.val(), self.embedding_sum.val());
        gather_codebook_embeddings(codebook, token_ids)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderVectorQuantization<B> {
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self._codebook.forward(token_ids)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderResidualVectorQuantization<B> {
    pub fn forward(&self, token_ids: Vec<Tensor<B, 2, Int>>) -> Tensor<B, 3> {
        let mut embeddings = self
            .layers
            .iter()
            .zip(token_ids)
            .map(|(layer, token_ids)| layer.forward(token_ids));
        let first = embeddings
            .next()
            .expect("residual vector quantization requires at least one layer");
        embeddings.fold(first, |accumulator, embedding| accumulator + embedding)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B> {
    pub fn forward_decode(&self, token_ids: Vec<Tensor<B, 2, Int>>) -> Tensor<B, 3> {
        let hidden = self.vq.forward(token_ids);
        self.output_proj.forward(hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderQuantizer<B> {
    pub fn forward(
        &self,
        codec_ids: Tensor<B, 3, Int>,
        num_semantic_quantizers: usize,
    ) -> Tensor<B, 3> {
        let [batch, _num_quantizers, time_steps] = codec_ids.dims();
        let total_layers = self.rvq_first.vq.layers.len() + self.rvq_rest.vq.layers.len();
        let per_layer_tokens = codec_ids
            .chunk(total_layers, 1)
            .into_iter()
            .map(|token_ids| token_ids.reshape([batch, time_steps]))
            .collect::<Vec<_>>();
        let mut per_layer_tokens = per_layer_tokens;
        let acoustic_tokens = per_layer_tokens.split_off(num_semantic_quantizers);
        let semantic = self.rvq_first.forward_decode(per_layer_tokens);
        if acoustic_tokens.is_empty() {
            semantic
        } else {
            let acoustic = self.rvq_rest.forward_decode(acoustic_tokens);
            semantic + acoustic
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoder<B> {
    pub fn forward(
        &self,
        codec_ids: Tensor<B, 3, Int>,
        num_semantic_quantizers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.quantizer.forward(codec_ids, num_semantic_quantizers);
        let hidden = self.pre_conv.forward(hidden);
        let hidden = hidden.swap_dims(1, 2);
        let mut hidden = self
            .pre_transformer
            .forward(hidden, num_heads, num_kv_heads, head_dim, rope, None)
            .swap_dims(1, 2);

        for (transposed_conv, conv_next) in &self.upsample {
            hidden = transposed_conv.forward(hidden);
            hidden = conv_next.forward(hidden);
        }

        for layer in &self.decoder {
            hidden = layer.forward(hidden);
        }
        hidden.clamp_min(-1.0).clamp_max(1.0)
    }
}
