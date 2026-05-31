use std::path::Path;

use burn::config::Config;
use burn::module::Initializer;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{LayerNormConfig, LinearConfig, PaddingConfig1d, RmsNormConfig};
use burn::tensor::backend::Backend;

use crate::Qwen3TtsLoadError;
use crate::model::codec::network::Qwen3TtsAudioCodec;
use crate::model::codec::network::activation::{AudioCodecLayerScale, AudioCodecSnakeBeta};
use crate::model::codec::network::conv::{
    AudioCodecCausalConv1d, AudioCodecCausalTransConv1d, ConvPadMode,
};
use crate::model::codec::network::decoder::{
    Qwen3TtsAudioCodecDecoder, Qwen3TtsAudioCodecDecoderAttention,
    Qwen3TtsAudioCodecDecoderCodebook, Qwen3TtsAudioCodecDecoderMlp,
    Qwen3TtsAudioCodecDecoderQuantizer, Qwen3TtsAudioCodecDecoderResidualVectorQuantization,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantizer, Qwen3TtsAudioCodecDecoderTransformer,
    Qwen3TtsAudioCodecDecoderTransformerLayer, Qwen3TtsAudioCodecDecoderVectorQuantization,
};
use crate::model::codec::network::encoder::{
    Qwen3TtsAudioCodecEncoder, Qwen3TtsAudioCodecEncoderActivation,
    Qwen3TtsAudioCodecEncoderAttention, Qwen3TtsAudioCodecEncoderBackbone,
    Qwen3TtsAudioCodecEncoderBackboneLayer, Qwen3TtsAudioCodecEncoderCodebook,
    Qwen3TtsAudioCodecEncoderConvLayer, Qwen3TtsAudioCodecEncoderMlp,
    Qwen3TtsAudioCodecEncoderQuantizer, Qwen3TtsAudioCodecEncoderResidualVectorQuantizer,
    Qwen3TtsAudioCodecEncoderResnetLayer, Qwen3TtsAudioCodecEncoderTransformer,
    Qwen3TtsAudioCodecEncoderTransformerLayer, Qwen3TtsAudioCodecEncoderVectorQuantization,
};
use crate::model::codec::network::wave::{
    Qwen3TtsAudioCodecConvNeXtBlock, Qwen3TtsAudioCodecWaveDecoderConvEntry,
    Qwen3TtsAudioCodecWaveDecoderEntry, Qwen3TtsAudioCodecWaveDecoderResidualUnit,
    Qwen3TtsAudioCodecWaveDecoderUpsampleStage,
};

#[derive(Config, Debug)]
pub struct Qwen3TtsAudioCodecConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_config: Qwen3TtsAudioCodecEncoderConfig,
    pub decoder_config: Qwen3TtsAudioCodecDecoderConfig,
    pub transformers_version: String,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsAudioCodecDecoderConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub semantic_codebook_size: usize,
    pub sliding_window: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsAudioCodecEncoderConfig {
    pub _frame_rate: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub audio_channels: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub compress: usize,
    pub dilation_growth_rate: usize,
    pub dtype: String,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub normalize: bool,
    pub num_attention_heads: usize,
    pub num_filters: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_residual_layers: usize,
    pub num_semantic_quantizers: usize,
    pub pad_mode: String,
    pub residual_kernel_size: usize,
    pub rope_theta: f64,
    pub sampling_rate: usize,
    pub sliding_window: usize,
    pub transformers_version: String,
    pub trim_right_ratio: f64,
    pub upsample_groups: usize,
    pub upsampling_ratios: Vec<usize>,
    pub use_cache: bool,
    pub use_causal_conv: bool,
    pub use_conv_shortcut: bool,
    pub use_streaming: bool,
    pub vector_quantization_hidden_dimension: usize,
}

impl Qwen3TtsAudioCodecConfig {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = path.as_ref().to_path_buf();
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }

    pub fn init_model<B: Backend>(&self, device: &B::Device) -> Qwen3TtsAudioCodec<B> {
        Qwen3TtsAudioCodec {
            decoder: self.decoder_config.init(device),
            encoder: self.init_encoder(device),
        }
    }

    fn init_encoder<B: Backend>(&self, device: &B::Device) -> Qwen3TtsAudioCodecEncoder<B> {
        let downsample_factor = derive_encoder_downsample_factor(
            self.input_sample_rate,
            self.encode_downsample_rate,
            self.encoder_config.sampling_rate,
            &self.encoder_config.upsampling_ratios,
            self.encoder_config._frame_rate,
        );
        let downsample_kernel = derive_encoder_downsample_kernel(downsample_factor);

        Qwen3TtsAudioCodecEncoder {
            encoder: self.encoder_config.init_backbone::<B>(device),
            encoder_transformer: self.encoder_config.init_transformer::<B>(device),
            downsample: AudioCodecCausalConv1d::<B>::new(
                self.encoder_config.hidden_size,
                self.encoder_config.hidden_size,
                downsample_kernel,
                2,
                1,
                1,
                false,
                ConvPadMode::Replicate,
                device,
            ),
            quantizer: self.encoder_config.init_quantizer::<B>(device),
        }
    }
}

fn derive_encoder_downsample_factor(
    input_sample_rate: usize,
    encode_downsample_rate: usize,
    encoder_sampling_rate: usize,
    upsampling_ratios: &[usize],
    frame_rate: f64,
) -> usize {
    let computed_frame_rate = input_sample_rate as f64 / encode_downsample_rate as f64;
    debug_assert!((computed_frame_rate - frame_rate).abs() < 1e-6);
    let backbone_frame_rate =
        encoder_sampling_rate as f64 / upsampling_ratios.iter().copied().product::<usize>() as f64;
    (backbone_frame_rate / frame_rate).round() as usize
}

fn derive_encoder_downsample_kernel(downsample_factor: usize) -> usize {
    downsample_factor * 2
}

impl Qwen3TtsAudioCodecEncoderConfig {
    fn init_backbone<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderBackbone<B> {
        let mut layers = Vec::with_capacity(self.upsampling_ratios.len() * 3 + 3);
        layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(
            Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(self.audio_channels, self.num_filters, self.kernel_size)
                    .with_bias(true)
                    .init(device),
                pad_mode: ConvPadMode::Constant,
            },
        ));

        let mut scaling = 1;
        for ratio in self.upsampling_ratios.iter().rev().copied() {
            let current_scale = scaling * self.num_filters;
            let hidden = current_scale / self.compress;
            layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::Resnet(
                Qwen3TtsAudioCodecEncoderResnetLayer {
                    conv_in: AudioCodecCausalConv1d::<B>::new(
                        current_scale,
                        hidden,
                        self.residual_kernel_size,
                        1,
                        1,
                        1,
                        true,
                        ConvPadMode::Constant,
                        device,
                    ),
                    conv_out: AudioCodecCausalConv1d::<B>::new(
                        hidden,
                        current_scale,
                        1,
                        1,
                        1,
                        1,
                        true,
                        ConvPadMode::Constant,
                        device,
                    ),
                },
            ));

            layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::Activation(
                Qwen3TtsAudioCodecEncoderActivation,
            ));
            layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(
                Qwen3TtsAudioCodecEncoderConvLayer {
                    conv: Conv1dConfig::new(current_scale, current_scale * 2, ratio * 2)
                        .with_stride(ratio)
                        .with_bias(true)
                        .init(device),
                    pad_mode: ConvPadMode::Constant,
                },
            ));
            scaling *= 2;
        }

        layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::Activation(
            Qwen3TtsAudioCodecEncoderActivation,
        ));
        layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(
            Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(
                    scaling * self.num_filters,
                    self.hidden_size,
                    self.last_kernel_size,
                )
                .with_bias(true)
                .init(device),
                pad_mode: ConvPadMode::Constant,
            },
        ));

        Qwen3TtsAudioCodecEncoderBackbone { layers }
    }

    fn init_transformer<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderTransformer<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsAudioCodecEncoderTransformer {
            layers: (0..self.num_hidden_layers)
                .map(|_| Qwen3TtsAudioCodecEncoderTransformerLayer {
                    self_attn: Qwen3TtsAudioCodecEncoderAttention {
                        q_proj: LinearConfig::new(self.hidden_size, q_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        k_proj: LinearConfig::new(self.hidden_size, kv_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        v_proj: LinearConfig::new(self.hidden_size, kv_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        o_proj: LinearConfig::new(q_out, self.hidden_size)
                            .with_bias(self.attention_bias)
                            .init(device),
                    },
                    mlp: Qwen3TtsAudioCodecEncoderMlp {
                        fc1: LinearConfig::new(self.hidden_size, self.intermediate_size)
                            .with_bias(false)
                            .init(device),
                        fc2: LinearConfig::new(self.intermediate_size, self.hidden_size)
                            .with_bias(false)
                            .init(device),
                    },
                    input_layernorm: LayerNormConfig::new(self.hidden_size)
                        .with_epsilon(self.norm_eps)
                        .with_bias(true)
                        .init(device),
                    post_attention_layernorm: LayerNormConfig::new(self.hidden_size)
                        .with_epsilon(self.norm_eps)
                        .with_bias(true)
                        .init(device),
                    self_attn_layer_scale: AudioCodecLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                    mlp_layer_scale: AudioCodecLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                })
                .collect(),
        }
    }

    fn init_quantizer<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderQuantizer<B> {
        Qwen3TtsAudioCodecEncoderQuantizer {
            semantic_residual_vector_quantizer: self
                .init_encoder_rvq(self.num_semantic_quantizers, device),
            acoustic_residual_vector_quantizer: self
                .init_encoder_rvq(self.num_quantizers - self.num_semantic_quantizers, device),
        }
    }

    fn init_encoder_rvq<B: Backend>(
        &self,
        num_layers: usize,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B> {
        Qwen3TtsAudioCodecEncoderResidualVectorQuantizer {
            input_proj: Conv1dConfig::new(
                self.hidden_size,
                self.vector_quantization_hidden_dimension,
                1,
            )
            .with_bias(false)
            .init(device),
            output_proj: Conv1dConfig::new(
                self.vector_quantization_hidden_dimension,
                self.hidden_size,
                1,
            )
            .with_bias(false)
            .init(device),
            layers: (0..num_layers)
                .map(|_| Qwen3TtsAudioCodecEncoderVectorQuantization {
                    codebook: Qwen3TtsAudioCodecEncoderCodebook::new(
                        self.codebook_size,
                        self.codebook_dim,
                        device,
                    ),
                })
                .collect(),
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecEncoderCodebook<B> {
    fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            initialized: Initializer::Ones.init([1], device),
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embed_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}

impl Qwen3TtsAudioCodecDecoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TtsAudioCodecDecoder<B> {
        let mut wave_decoder = Vec::with_capacity(self.upsample_rates.len() + 3);
        wave_decoder.push(Qwen3TtsAudioCodecWaveDecoderEntry::InputConv(
            Qwen3TtsAudioCodecWaveDecoderConvEntry {
                conv: Conv1dConfig::new(self.latent_dim, self.decoder_dim, 7)
                    .with_bias(true)
                    .with_padding(PaddingConfig1d::Explicit(6, 0))
                    .init(device),
            },
        ));

        for layer_idx in 0..self.upsample_rates.len() {
            wave_decoder.push(self.init_wave_decoder_block(layer_idx, device));
        }

        let output_dim = self.decoder_dim / (1 << self.upsample_rates.len());
        wave_decoder.push(Qwen3TtsAudioCodecWaveDecoderEntry::OutputActivation(
            AudioCodecSnakeBeta::<B>::new(output_dim, device),
        ));
        wave_decoder.push(Qwen3TtsAudioCodecWaveDecoderEntry::OutputConv(
            Qwen3TtsAudioCodecWaveDecoderConvEntry {
                conv: Conv1dConfig::new(output_dim, 1, 7)
                    .with_bias(true)
                    .with_padding(PaddingConfig1d::Explicit(6, 0))
                    .init(device),
            },
        ));

        Qwen3TtsAudioCodecDecoder {
            pre_transformer: self.init_pre_transformer(device),
            quantizer: self.init_quantizer(device),
            pre_conv: AudioCodecCausalConv1d::<B>::new(
                self.codebook_dim,
                self.latent_dim,
                3,
                1,
                1,
                1,
                true,
                ConvPadMode::Constant,
                device,
            ),
            upsample: self
                .upsampling_ratios
                .iter()
                .map(|&ratio| {
                    (
                        AudioCodecCausalTransConv1d::<B>::new(
                            self.latent_dim,
                            self.latent_dim,
                            ratio,
                            ratio,
                            1,
                            true,
                            device,
                        ),
                        Qwen3TtsAudioCodecConvNeXtBlock {
                            dwconv: AudioCodecCausalConv1d::<B>::new(
                                self.latent_dim,
                                self.latent_dim,
                                7,
                                1,
                                1,
                                self.latent_dim,
                                true,
                                ConvPadMode::Constant,
                                device,
                            ),
                            norm: LayerNormConfig::new(self.latent_dim)
                                .with_epsilon(1e-6)
                                .with_bias(true)
                                .init(device),
                            pwconv1: LinearConfig::new(self.latent_dim, self.latent_dim * 4)
                                .with_bias(true)
                                .init(device),
                            pwconv2: LinearConfig::new(self.latent_dim * 4, self.latent_dim)
                                .with_bias(true)
                                .init(device),
                            gamma: Initializer::Constant { value: 1e-6 }
                                .init([self.latent_dim], device),
                        },
                    )
                })
                .collect(),
            decoder: wave_decoder,
        }
    }

    fn init_pre_transformer<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecDecoderTransformer<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsAudioCodecDecoderTransformer {
            layers: (0..self.num_hidden_layers)
                .map(|_| Qwen3TtsAudioCodecDecoderTransformerLayer {
                    self_attn: Qwen3TtsAudioCodecDecoderAttention {
                        q_proj: LinearConfig::new(self.hidden_size, q_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        k_proj: LinearConfig::new(self.hidden_size, kv_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        v_proj: LinearConfig::new(self.hidden_size, kv_out)
                            .with_bias(self.attention_bias)
                            .init(device),
                        o_proj: LinearConfig::new(q_out, self.hidden_size)
                            .with_bias(self.attention_bias)
                            .init(device),
                    },
                    mlp: Qwen3TtsAudioCodecDecoderMlp {
                        gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                            .with_bias(false)
                            .init(device),
                        up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                            .with_bias(false)
                            .init(device),
                        down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                            .with_bias(false)
                            .init(device),
                    },
                    input_layernorm: RmsNormConfig::new(self.hidden_size)
                        .with_epsilon(self.rms_norm_eps)
                        .init(device),
                    post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                        .with_epsilon(self.rms_norm_eps)
                        .init(device),
                    self_attn_layer_scale: AudioCodecLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                    mlp_layer_scale: AudioCodecLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                })
                .collect(),
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            input_proj: LinearConfig::new(self.latent_dim, self.hidden_size)
                .with_bias(true)
                .init(device),
            output_proj: LinearConfig::new(self.hidden_size, self.latent_dim)
                .with_bias(true)
                .init(device),
        }
    }

    fn init_quantizer<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecDecoderQuantizer<B> {
        let hidden = self.codebook_dim / 2;

        Qwen3TtsAudioCodecDecoderQuantizer {
            rvq_first: Qwen3TtsAudioCodecDecoderResidualVectorQuantizer {
                input_proj: Conv1dConfig::new(self.codebook_dim, hidden, 1)
                    .with_bias(false)
                    .init(device),
                output_proj: Conv1dConfig::new(hidden, self.codebook_dim, 1)
                    .with_bias(false)
                    .init(device),
                vq: Qwen3TtsAudioCodecDecoderResidualVectorQuantization {
                    layers: vec![Qwen3TtsAudioCodecDecoderVectorQuantization {
                        _codebook: Qwen3TtsAudioCodecDecoderCodebook::new(
                            self.codebook_size,
                            hidden,
                            device,
                        ),
                    }],
                },
            },
            rvq_rest: Qwen3TtsAudioCodecDecoderResidualVectorQuantizer {
                input_proj: Conv1dConfig::new(self.codebook_dim, hidden, 1)
                    .with_bias(false)
                    .init(device),
                output_proj: Conv1dConfig::new(hidden, self.codebook_dim, 1)
                    .with_bias(false)
                    .init(device),
                vq: Qwen3TtsAudioCodecDecoderResidualVectorQuantization {
                    layers: (0..self
                        .num_quantizers
                        .saturating_sub(self.num_semantic_quantizers))
                        .map(|_| Qwen3TtsAudioCodecDecoderVectorQuantization {
                            _codebook: Qwen3TtsAudioCodecDecoderCodebook::new(
                                self.codebook_size,
                                hidden,
                                device,
                            ),
                        })
                        .collect(),
                },
            },
        }
    }

    fn init_wave_decoder_block<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecWaveDecoderEntry<B> {
        let in_dim = self.decoder_dim / (1 << layer_idx);
        let out_dim = self.decoder_dim / (1 << (layer_idx + 1));
        let upsample_rate = self.upsample_rates[layer_idx];

        Qwen3TtsAudioCodecWaveDecoderEntry::UpsampleStage(
            Qwen3TtsAudioCodecWaveDecoderUpsampleStage {
                block: (
                    AudioCodecSnakeBeta::<B>::new(in_dim, device),
                    AudioCodecCausalTransConv1d::<B>::new(
                        in_dim,
                        out_dim,
                        upsample_rate * 2,
                        upsample_rate,
                        1,
                        true,
                        device,
                    ),
                    self.init_wave_decoder_residual_unit(out_dim, 1, device),
                    self.init_wave_decoder_residual_unit(out_dim, 3, device),
                    self.init_wave_decoder_residual_unit(out_dim, 9, device),
                ),
            },
        )
    }

    fn init_wave_decoder_residual_unit<B: Backend>(
        &self,
        channels: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecWaveDecoderResidualUnit<B> {
        Qwen3TtsAudioCodecWaveDecoderResidualUnit {
            act1: AudioCodecSnakeBeta::<B>::new(channels, device),
            conv1: AudioCodecCausalConv1d::<B>::new(
                channels,
                channels,
                7,
                1,
                dilation,
                1,
                true,
                ConvPadMode::Constant,
                device,
            ),
            act2: AudioCodecSnakeBeta::<B>::new(channels, device),
            conv2: AudioCodecCausalConv1d::<B>::new(
                channels,
                channels,
                1,
                1,
                1,
                1,
                true,
                ConvPadMode::Constant,
                device,
            ),
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderCodebook<B> {
    fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embedding_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}
