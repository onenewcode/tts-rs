use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear, RotaryEncoding, RotaryEncodingConfig};
use burn::tensor::activation::{elu, gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use super::activation::AudioCodecLayerScale;
use super::conv::{AudioCodecCausalConv1d, ConvPadMode, forward_padded_conv1d};
use crate::Qwen3TtsInferenceError;
use crate::model::codec::config::Qwen3TtsAudioCodecEncoderConfig;
use crate::model::nn::attention::{autoregressive_attention_mask, repeat_kv_heads};
use crate::model::nn::codebook::{
    gather_codebook_embeddings, nearest_codebook_token_ids, normalized_codebook_centroids,
};
use crate::model::nn::tensor::{
    flatten_batch_sequence, read_int_tensor_vec, unflatten_batch_sequence,
};

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoder<B: Backend> {
    pub encoder: Qwen3TtsAudioCodecEncoderBackbone<B>,
    pub encoder_transformer: Qwen3TtsAudioCodecEncoderTransformer<B>,
    pub downsample: AudioCodecCausalConv1d<B>,
    pub quantizer: Qwen3TtsAudioCodecEncoderQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderBackbone<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecEncoderBackboneLayer<B>>,
}

#[derive(Module, Debug, Clone)]
pub struct Qwen3TtsAudioCodecEncoderActivation;

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3TtsAudioCodecEncoderBackboneLayer<B: Backend> {
    InputConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
    Resnet(Qwen3TtsAudioCodecEncoderResnetLayer<B>),
    Activation(Qwen3TtsAudioCodecEncoderActivation),
    DownsampleConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
    OutputConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderConvLayer<B: Backend> {
    pub conv: Conv1d<B>,
    #[module(skip)]
    pub pad_mode: ConvPadMode,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderResnetLayer<B: Backend> {
    pub conv_in: AudioCodecCausalConv1d<B>,
    pub conv_out: AudioCodecCausalConv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecEncoderTransformerLayer<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsAudioCodecEncoderAttention<B>,
    pub mlp: Qwen3TtsAudioCodecEncoderMlp<B>,
    pub input_layernorm: LayerNorm<B>,
    pub post_attention_layernorm: LayerNorm<B>,
    pub self_attn_layer_scale: AudioCodecLayerScale<B>,
    pub mlp_layer_scale: AudioCodecLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderQuantizer<B: Backend> {
    pub semantic_residual_vector_quantizer: Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B>,
    pub acoustic_residual_vector_quantizer: Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub layers: Vec<Qwen3TtsAudioCodecEncoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderVectorQuantization<B: Backend> {
    pub codebook: Qwen3TtsAudioCodecEncoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderCodebook<B: Backend> {
    pub initialized: Param<Tensor<B, 1>>,
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embed_sum: Param<Tensor<B, 2>>,
}

impl<B: Backend> Qwen3TtsAudioCodecEncoder<B> {
    pub fn encode_reference_frames(
        &self,
        config: &Qwen3TtsAudioCodecEncoderConfig,
        valid_num_quantizers: usize,
        waveform: Tensor<B, 3>,
    ) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
        let encoded = self.encoder.forward(waveform);
        let transformed = self.encoder_transformer.forward(encoded, config);
        let downsampled = self.downsample.forward(transformed);
        self.quantizer.extract_reference_frames(
            downsampled,
            config.num_semantic_quantizers,
            valid_num_quantizers,
        )
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderBackbone<B> {
    pub fn forward(&self, mut hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            hidden = match layer {
                Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(layer) => layer.forward(hidden),
                Qwen3TtsAudioCodecEncoderBackboneLayer::Resnet(layer) => layer.forward(hidden),
                Qwen3TtsAudioCodecEncoderBackboneLayer::Activation(layer) => layer.forward(hidden),
                Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(layer)
                | Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(layer) => {
                    layer.forward(hidden)
                }
            };
        }
        hidden
    }
}

impl Qwen3TtsAudioCodecEncoderActivation {
    pub fn forward<B: Backend>(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        elu(hidden, 1.0)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderConvLayer<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        forward_padded_conv1d(&self.conv, self.pad_mode, hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderResnetLayer<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.conv_in.forward(elu(hidden, 1.0));
        let hidden = elu(hidden, 1.0);
        let hidden = self.conv_out.forward(hidden);
        residual + hidden
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderTransformer<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        config: &Qwen3TtsAudioCodecEncoderConfig,
    ) -> Tensor<B, 3> {
        let rope = RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim)
            .with_theta(config.rope_theta as f32)
            .init(&hidden.device());
        let mut hidden = hidden.swap_dims(1, 2);
        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                &rope,
            );
        }
        hidden.swap_dims(1, 2)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderTransformerLayer<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.input_layernorm.forward(hidden);
        let hidden = self
            .self_attn
            .forward(hidden, num_heads, num_kv_heads, head_dim, rope);
        let hidden = self.self_attn_layer_scale.forward(hidden);
        let hidden = residual + hidden;

        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(hidden);
        let hidden = self.mlp.forward(hidden);
        let hidden = self.mlp_layer_scale.forward(hidden);
        residual + hidden
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderAttention<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = hidden.dims();
        let device = hidden.device();
        let hidden_2d = flatten_batch_sequence(hidden);
        let query = self
            .q_proj
            .forward(hidden_2d.clone())
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(hidden_2d.clone())
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(hidden_2d)
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);

        let query = rope.apply(query, 0);
        let key = rope.apply(key, 0);
        let key = repeat_kv_heads(key, num_heads / num_kv_heads);
        let value = repeat_kv_heads(value, num_heads / num_kv_heads);

        let dtype = query.dtype();
        let attention_scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt())
            .mask_fill(
                autoregressive_attention_mask::<B>(batch_size, seq_len, &device),
                f32::NEG_INFINITY,
            );
        let attention_weights = softmax(attention_scores.cast(DType::F32), 3).cast(dtype);
        let attention_output = attention_weights.matmul(value);
        let attention_output =
            attention_output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);

        let output = self
            .o_proj
            .forward(flatten_batch_sequence(attention_output));
        unflatten_batch_sequence(output, batch_size, seq_len)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderMlp<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = hidden.dims();
        let hidden_2d = flatten_batch_sequence(hidden);
        let hidden = self.fc1.forward(hidden_2d);
        let hidden = gelu(hidden);
        let hidden = self.fc2.forward(hidden);
        unflatten_batch_sequence(hidden, batch_size, seq_len)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderQuantizer<B> {
    pub fn extract_reference_frames(
        &self,
        hidden: Tensor<B, 3>,
        semantic_layers: usize,
        valid_layers: usize,
    ) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
        let acoustic_layers = valid_layers.saturating_sub(semantic_layers);
        let semantic_codes = self
            .semantic_residual_vector_quantizer
            .encode(hidden.clone(), semantic_layers)?;
        let acoustic_codes = self
            .acoustic_residual_vector_quantizer
            .encode(hidden, acoustic_layers)?;

        if semantic_codes.is_empty() {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: "reference audio produced no semantic codec frames".to_string(),
            });
        }

        let time_steps = semantic_codes[0].dims()[1];
        let all_codes: Vec<Tensor<B, 2, Int>> =
            semantic_codes.into_iter().chain(acoustic_codes).collect();
        let total_layers = all_codes.len();
        let flat_codes = read_int_tensor_vec(
            Tensor::cat(all_codes, 0),
            "failed to read reference codec token ids",
        )?;

        let mut frames = Vec::with_capacity(time_steps);
        for time_index in 0..time_steps {
            let mut frame = Vec::with_capacity(valid_layers);
            for layer_idx in 0..total_layers {
                frame.push(flat_codes[layer_idx * time_steps + time_index]);
            }
            frames.push(frame);
        }
        Ok(frames)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B> {
    pub fn encode(
        &self,
        hidden: Tensor<B, 3>,
        max_layers: usize,
    ) -> Result<Vec<Tensor<B, 2, Int>>, Qwen3TtsInferenceError> {
        if max_layers == 0 {
            return Ok(Vec::new());
        }

        let projected = self.input_proj.forward(hidden);
        let mut residual = projected.clone();
        let mut all_codes = Vec::with_capacity(max_layers);
        for layer in self.layers.iter().take(max_layers) {
            let (codes, quantized) = layer.nearest_tokens_and_quantized(residual.clone())?;
            residual = residual - quantized;
            all_codes.push(codes);
        }
        Ok(all_codes)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderVectorQuantization<B> {
    pub fn nearest_tokens_and_quantized(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 2, Int>, Tensor<B, 3>), Qwen3TtsInferenceError> {
        let [batch_size, hidden_size, time_steps] = hidden.dims();
        if batch_size != 1 || hidden_size == 0 || time_steps == 0 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "semantic quantizer expects [1, hidden, time] with non-zero dims, got [{batch_size}, {hidden_size}, {time_steps}]"
                ),
            });
        }

        let hidden_dtype = hidden.dtype();
        let codebook_size = self.codebook.cluster_usage.dims()[0];
        if codebook_size == 0 || self.codebook.embed_sum.dims() != [codebook_size, hidden_size] {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: "semantic codebook tensor shapes are inconsistent".to_string(),
            });
        }

        let centroids = normalized_codebook_centroids(
            self.codebook.cluster_usage.val(),
            self.codebook.embed_sum.val(),
        );
        let token_ids = nearest_codebook_token_ids(hidden, centroids.clone());
        let quantized = gather_codebook_embeddings(centroids, token_ids.clone()).cast(hidden_dtype);
        Ok((token_ids, quantized))
    }
}
