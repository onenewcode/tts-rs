use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear, RmsNorm, RotaryEncoding, RotaryEncodingConfig};
use burn::tensor::activation::{elu, gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use super::activation::{AudioCodecLayerScale, AudioCodecSnakeBeta};
use super::conv::{
    AudioCodecCausalConv1d, AudioCodecCausalTransConv1d, ConvPadMode, forward_padded_conv1d,
};
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
pub struct Qwen3TtsAudioCodec<B: Backend> {
    pub decoder: Qwen3TtsAudioCodecDecoder<B>,
    pub encoder: Qwen3TtsAudioCodecEncoder<B>,
}

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

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecDecoder<B: Backend> {
    pub pre_transformer: Qwen3TtsAudioCodecDecoderTransformer<B>,
    pub quantizer: Qwen3TtsAudioCodecDecoderQuantizer<B>,
    pub pre_conv: AudioCodecCausalConv1d<B>,
    pub upsample: Vec<(
        AudioCodecCausalTransConv1d<B>,
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

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3TtsAudioCodecWaveDecoderEntry<B: Backend> {
    InputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
    UpsampleStage(Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B>),
    OutputActivation(AudioCodecSnakeBeta<B>),
    OutputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecWaveDecoderConvEntry<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
#[allow(clippy::type_complexity)]
pub struct Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B: Backend> {
    pub block: (
        AudioCodecSnakeBeta<B>,
        AudioCodecCausalTransConv1d<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecWaveDecoderResidualUnit<B: Backend> {
    pub act1: AudioCodecSnakeBeta<B>,
    pub conv1: AudioCodecCausalConv1d<B>,
    pub act2: AudioCodecSnakeBeta<B>,
    pub conv2: AudioCodecCausalConv1d<B>,
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

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderConvEntry<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderResidualUnit<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.act1.forward(hidden);
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        let hidden = self.conv2.forward(hidden);
        residual + hidden
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.block.0.forward(hidden);
        let hidden = self.block.1.forward(hidden);
        let hidden = self.block.2.forward(hidden);
        let hidden = self.block.3.forward(hidden);
        self.block.4.forward(hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderEntry<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::InputConv(entry) => entry.forward(hidden),
            Self::UpsampleStage(stage) => stage.forward(hidden),
            Self::OutputActivation(snake) => snake.forward(hidden),
            Self::OutputConv(entry) => entry.forward(hidden),
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecConvNeXtBlock<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let [batch, channels, time] = hidden.dims();
        let hidden = self.dwconv.forward(hidden);
        let hidden = flatten_batch_sequence(hidden.swap_dims(1, 2));
        let hidden = self.norm.forward(hidden);
        let hidden = unflatten_batch_sequence(hidden, batch, time).swap_dims(1, 2);
        let hidden = flatten_batch_sequence(hidden.swap_dims(1, 2));
        let hidden = self.pwconv1.forward(hidden);
        let hidden = gelu(hidden);
        let hidden = self.pwconv2.forward(hidden);
        let hidden = unflatten_batch_sequence(hidden, batch, time).swap_dims(1, 2);
        let gamma = self.gamma.val().reshape([1, channels, 1]);
        residual + hidden.mul(gamma)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderMlp<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden_size] = hidden.dims();
        let hidden_2d = flatten_batch_sequence(hidden);
        let gate = self.gate_proj.forward(hidden_2d.clone());
        let up = self.up_proj.forward(hidden_2d);
        let activated = silu(gate);
        let product = activated * up;
        let output = self.down_proj.forward(product);
        unflatten_batch_sequence(output, batch, seq_len)
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
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = hidden.dims();
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
            .div_scalar((head_dim as f32).sqrt());
        let attention_scores = if let Some(mask) = mask {
            attention_scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attention_scores
        };
        let attention_weights = softmax(attention_scores.cast(DType::F32), 3).cast(dtype);
        let attention_output = attention_weights.matmul(value);
        let output =
            attention_output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);

        let output = self.o_proj.forward(flatten_batch_sequence(output));
        unflatten_batch_sequence(output, batch_size, seq_len)
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
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
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
        mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _latent] = hidden.dims();
        let hidden_2d = flatten_batch_sequence(hidden);
        let hidden = self.input_proj.forward(hidden_2d);
        let mut hidden = unflatten_batch_sequence(hidden, batch, seq_len);

        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                rope,
                mask.clone(),
            );
        }

        let hidden = self.norm.forward(hidden);
        let hidden = self.output_proj.forward(flatten_batch_sequence(hidden));
        unflatten_batch_sequence(hidden, batch, seq_len)
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
    pub fn forward(&self, token_ids: &[Tensor<B, 2, Int>]) -> Tensor<B, 3> {
        let mut output: Option<Tensor<B, 3>> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let embedding = layer.forward(token_ids[layer_idx].clone());
            output = Some(match output {
                Some(accumulator) => accumulator + embedding,
                None => embedding,
            });
        }
        output.expect("residual vector quantization requires at least one layer")
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderResidualVectorQuantizer<B> {
    pub fn forward_decode(&self, token_ids: &[Tensor<B, 2, Int>]) -> Tensor<B, 3> {
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
        let per_layer_tokens: Vec<Tensor<B, 2, Int>> = (0..total_layers)
            .map(|layer_idx| {
                codec_ids
                    .clone()
                    .slice([0..batch, layer_idx..layer_idx + 1, 0..time_steps])
                    .reshape([batch, time_steps])
            })
            .collect();

        let semantic_tokens: &[Tensor<B, 2, Int>] = &per_layer_tokens[..num_semantic_quantizers];
        let acoustic_tokens: &[Tensor<B, 2, Int>] = &per_layer_tokens[num_semantic_quantizers..];

        let semantic = self.rvq_first.forward_decode(semantic_tokens);
        let acoustic = self.rvq_rest.forward_decode(acoustic_tokens);
        semantic + acoustic
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
