use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear, RmsNorm, RotaryEncoding, RotaryEncodingConfig};
use burn::tensor::activation::{elu, gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn::tensor::{DType, Int, Tensor, TensorData};

use super::activation::{AudioCodecLayerScale, AudioCodecSnakeBeta};
use super::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d};
use crate::model::codec::config::Qwen3TtsAudioCodecEncoderConfig;
use crate::Qwen3TtsInferenceError;

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
        let downsampled =
            streamable_conv1d(&self.downsample.conv, transformed, ConvPadMode::Replicate);
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
                Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(layer) => {
                    streamable_conv1d(&layer.conv, hidden, ConvPadMode::Constant)
                }
                Qwen3TtsAudioCodecEncoderBackboneLayer::Resnet(layer) => layer.forward(hidden),
                Qwen3TtsAudioCodecEncoderBackboneLayer::Activation(layer) => layer.forward(hidden),
                Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(layer)
                | Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(layer) => {
                    streamable_conv1d(&layer.conv, hidden, ConvPadMode::Constant)
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
        let hidden = layer_norm_3d(&self.input_layernorm, hidden);
        let hidden = self
            .self_attn
            .forward(hidden, num_heads, num_kv_heads, head_dim, rope);
        let hidden = self.self_attn_layer_scale.forward(hidden);
        let hidden = residual + hidden;

        let residual = hidden.clone();
        let hidden = layer_norm_3d(&self.post_attention_layernorm, hidden);
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
        let [batch_size, seq_len, hidden_size] = hidden.dims();
        let device = hidden.device();
        let hidden_2d = hidden.reshape([batch_size * seq_len, hidden_size]);
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
        let key = repeat_kv_encoder(key, num_heads / num_kv_heads);
        let value = repeat_kv_encoder(value, num_heads / num_kv_heads);

        let dtype = query.dtype();
        let attention_scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt())
            + causal_attention_bias(batch_size, num_heads, seq_len, dtype, &device);
        let attention_weights = softmax(attention_scores.cast(DType::F32), 3).cast(dtype);
        let attention_output = attention_weights.matmul(value);
        let attention_output =
            attention_output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);

        self.o_proj
            .forward(attention_output.reshape([batch_size * seq_len, num_heads * head_dim]))
            .reshape([batch_size, seq_len, hidden_size])
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderMlp<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden.dims();
        let hidden_2d = hidden.reshape([batch_size * seq_len, hidden_size]);
        let hidden = self.fc1.forward(hidden_2d);
        let hidden = gelu(hidden);
        self.fc2
            .forward(hidden)
            .reshape([batch_size, seq_len, hidden_size])
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

        let time_steps = semantic_codes[0].len();
        let mut frames = Vec::with_capacity(time_steps);
        for time_index in 0..time_steps {
            let mut frame = Vec::with_capacity(valid_layers);
            for layer_codes in &semantic_codes {
                frame.push(layer_codes[time_index]);
            }
            for layer_codes in &acoustic_codes {
                frame.push(layer_codes[time_index]);
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
    ) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
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
    ) -> Result<(Vec<i64>, Tensor<B, 3>), Qwen3TtsInferenceError> {
        let [batch_size, hidden_size, time_steps] = hidden.dims();
        if batch_size != 1 || hidden_size == 0 || time_steps == 0 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "semantic quantizer expects [1, hidden, time] with non-zero dims, got [{batch_size}, {hidden_size}, {time_steps}]"
                ),
            });
        }

        let hidden_values = hidden
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|source| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read semantic encoder activations: {source}"),
            })?;
        let cluster_usage = self
            .codebook
            .cluster_usage
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|source| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read semantic codebook usage: {source}"),
            })?;
        let embed_sum = self
            .codebook
            .embed_sum
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|source| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read semantic codebook embeddings: {source}"),
            })?;

        let codebook_size = cluster_usage.len();
        if codebook_size == 0 || embed_sum.len() != codebook_size * hidden_size {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: "semantic codebook tensor shapes are inconsistent".to_string(),
            });
        }

        let mut tokens = Vec::with_capacity(time_steps);
        let mut quantized_values = vec![0.0f32; hidden_size * time_steps];
        for time_index in 0..time_steps {
            let mut best_index = 0usize;
            let mut best_distance = f32::INFINITY;

            for code_index in 0..codebook_size {
                let usage = cluster_usage[code_index].max(1e-6);
                let mut distance = 0.0f32;
                for hidden_index in 0..hidden_size {
                    let hidden_value = hidden_values[hidden_index * time_steps + time_index];
                    let centroid = embed_sum[code_index * hidden_size + hidden_index] / usage;
                    let diff = hidden_value - centroid;
                    distance += diff * diff;
                }
                if distance < best_distance {
                    best_distance = distance;
                    best_index = code_index;
                }
            }

            tokens.push(best_index as i64);
            let usage = cluster_usage[best_index].max(1e-6);
            for hidden_index in 0..hidden_size {
                quantized_values[hidden_index * time_steps + time_index] =
                    embed_sum[best_index * hidden_size + hidden_index] / usage;
            }
        }

        let quantized = Tensor::<B, 3>::from_data(
            TensorData::new(quantized_values, [1, hidden_size, time_steps]),
            &hidden.device(),
        );
        Ok((tokens, quantized))
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
        let hidden = hidden.swap_dims(1, 2).reshape([batch * time, channels]);
        let hidden = self.norm.forward(hidden);
        let hidden = hidden.reshape([batch, time, channels]).swap_dims(1, 2);
        let hidden = hidden.swap_dims(1, 2).reshape([batch * time, channels]);
        let hidden = self.pwconv1.forward(hidden);
        let hidden = gelu(hidden);
        let hidden = self.pwconv2.forward(hidden);
        let hidden = hidden.reshape([batch, time, channels]).swap_dims(1, 2);
        let gamma = self.gamma.val().reshape([1, channels, 1]);
        residual + hidden.mul(gamma)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderMlp<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden_size] = hidden.dims();
        let hidden_2d = hidden.reshape([batch * seq_len, hidden_size]);
        let gate = self.gate_proj.forward(hidden_2d.clone());
        let up = self.up_proj.forward(hidden_2d);
        let activated = silu(gate);
        let product = activated * up;
        self.down_proj
            .forward(product)
            .reshape([batch, seq_len, hidden_size])
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
        let [batch_size, seq_len, hidden_size] = hidden.dims();
        let hidden_2d = hidden.reshape([batch_size * seq_len, hidden_size]);

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
        let key = repeat_kv_decoder(key, num_heads / num_kv_heads);
        let value = repeat_kv_decoder(value, num_heads / num_kv_heads);

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

        self.o_proj
            .forward(output.reshape([batch_size * seq_len, num_heads * head_dim]))
            .reshape([batch_size, seq_len, hidden_size])
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
        let [batch, seq_len, latent] = hidden.dims();
        let hidden_2d = hidden.reshape([batch * seq_len, latent]);
        let hidden = self.input_proj.forward(hidden_2d);
        let [_, hidden_size] = hidden.dims();
        let mut hidden = hidden.reshape([batch, seq_len, hidden_size]);

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
        let [batch, seq_len, hidden_size] = hidden.dims();
        self.output_proj
            .forward(hidden.reshape([batch * seq_len, hidden_size]))
            .reshape([batch, seq_len, latent])
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderCodebook<B> {
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = token_ids.dims();
        let [_codebook_size, embed_dim] = self.embedding_sum.dims();
        let usage = self
            .cluster_usage
            .val()
            .clamp_min(1e-6)
            .reshape([self.cluster_usage.dims()[0], 1]);
        let codebook = self.embedding_sum.val().div(usage);
        let gathered = codebook.select(0, token_ids.reshape([batch * seq_len]));
        gathered
            .reshape([batch, seq_len, embed_dim])
            .swap_dims(1, 2)
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

#[derive(Debug, Clone, Copy)]
enum ConvPadMode {
    Constant,
    Replicate,
}

fn streamable_conv1d<B: Backend>(
    conv: &Conv1d<B>,
    hidden: Tensor<B, 3>,
    pad_mode: ConvPadMode,
) -> Tensor<B, 3> {
    let time_steps = hidden.dims()[2];
    let effective_kernel = (conv.kernel_size - 1) * conv.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(conv.stride);
    let extra_padding =
        extra_padding_for_conv1d(time_steps, effective_kernel, conv.stride, padding_total);
    let hidden = pad_1d(hidden, padding_total, extra_padding, pad_mode);
    conv.forward(hidden)
}

fn extra_padding_for_conv1d(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let frame_count =
        (len + padding_total).saturating_sub(kernel_size) as f64 / stride as f64 + 1.0;
    let ideal_len = ((frame_count.ceil() as usize).saturating_sub(1) * stride + kernel_size)
        .saturating_sub(padding_total);
    ideal_len.saturating_sub(len)
}

fn pad_1d<B: Backend>(
    hidden: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
    mode: ConvPadMode,
) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return hidden;
    }
    match mode {
        ConvPadMode::Constant => hidden.pad((pad_left, pad_right, 0, 0), PadMode::Constant(0.0)),
        ConvPadMode::Replicate => replicate_pad_1d(hidden, pad_left, pad_right),
    }
}

fn replicate_pad_1d<B: Backend>(
    hidden: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
) -> Tensor<B, 3> {
    let [batch, channels, time] = hidden.dims();
    let mut segments = Vec::with_capacity(3);
    if pad_left > 0 {
        segments.push(
            hidden
                .clone()
                .slice([0..batch, 0..channels, 0..1])
                .repeat_dim(2, pad_left),
        );
    }
    segments.push(hidden.clone());
    if pad_right > 0 {
        segments.push(
            hidden
                .slice([0..batch, 0..channels, time - 1..time])
                .repeat_dim(2, pad_right),
        );
    }
    Tensor::cat(segments, 2)
}

fn causal_attention_bias<B: Backend>(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for query_idx in 0..seq_len {
        for key_idx in 0..seq_len {
            values.push(if key_idx > query_idx {
                f32::NEG_INFINITY
            } else {
                0.0
            });
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(values, [1, 1, seq_len, seq_len]), device)
        .repeat_dim(0, batch_size)
        .repeat_dim(1, num_heads)
        .cast(dtype)
}

fn layer_norm_3d<B: Backend>(norm: &LayerNorm<B>, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    norm.forward(hidden.reshape([batch_size * seq_len, hidden_size]))
        .reshape([batch_size, seq_len, hidden_size])
}

fn repeat_kv_encoder<B: Backend>(hidden: Tensor<B, 4>, repetitions: usize) -> Tensor<B, 4> {
    if repetitions == 1 {
        return hidden;
    }
    let [batch_size, num_heads, seq_len, head_dim] = hidden.dims();
    hidden
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repetitions)
        .reshape([batch_size, num_heads * repetitions, seq_len, head_dim])
}

fn repeat_kv_decoder<B: Backend>(hidden: Tensor<B, 4>, repetitions: usize) -> Tensor<B, 4> {
    if repetitions == 1 {
        return hidden;
    }
    let [batch_size, num_kv_heads, seq_len, head_dim] = hidden.dims();
    hidden
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repetitions)
        .reshape([batch_size, num_kv_heads * repetitions, seq_len, head_dim])
}
