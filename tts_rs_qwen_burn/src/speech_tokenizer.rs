use std::path::{Path, PathBuf};

use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{Tensor, backend::Backend};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::manifest::{
    LoadReport, VerificationArtifacts, WeightVerificationReport, verify_module_weights,
};
use crate::{Qwen3TtsLoadError, Qwen3TtsVerifyError};

#[derive(Config, Debug)]
pub struct Qwen3TtsSpeechTokenizerConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_config: Qwen3TtsSpeechTokenizerEncoderConfig,
    pub decoder_config: Qwen3TtsSpeechTokenizerDecoderConfig,
    pub transformers_version: String,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderConfig {
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
pub struct Qwen3TtsSpeechTokenizerEncoderConfig {
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

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerCheckpoint<B: Backend> {
    pub decoder: Qwen3TtsSpeechTokenizerDecoder<B>,
    pub encoder: Qwen3TtsSpeechTokenizerEncoder<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoder<B: Backend> {
    pub pre_transformer: Qwen3TtsSpeechTokenizerDecoderTransformer<B>,
    pub quantizer: Qwen3TtsSpeechTokenizerDecoderQuantizer<B>,
    pub pre_conv: TokenizerCausalConv1d<B>,
    pub upsample: Vec<(
        TokenizerCausalTransConv1d<B>,
        Qwen3TtsSpeechTokenizerConvNeXtBlock<B>,
    )>,
    pub decoder: Vec<Qwen3TtsSpeechTokenizerWaveDecoderEntry<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerConvNeXtBlock<B: Backend> {
    pub dwconv: TokenizerCausalConv1d<B>,
    pub norm: LayerNorm<B>,
    pub pwconv1: Linear<B>,
    pub pwconv2: Linear<B>,
    pub gamma: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoder<B: Backend> {
    pub encoder: Qwen3TtsSpeechTokenizerEncoderBackbone<B>,
    pub encoder_transformer: Qwen3TtsSpeechTokenizerEncoderTransformer<B>,
    pub downsample: TokenizerCausalConv1d<B>,
    pub quantizer: Qwen3TtsSpeechTokenizerEncoderQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerDecoderTransformerLayer<B>>,
    pub norm: RmsNorm<B>,
    pub input_proj: Linear<B>,
    pub output_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsSpeechTokenizerDecoderAttention<B>,
    pub mlp: Qwen3TtsSpeechTokenizerDecoderMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub self_attn_layer_scale: TokenizerLayerScale<B>,
    pub mlp_layer_scale: TokenizerLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderQuantizer<B: Backend> {
    pub rvq_first: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B>,
    pub rvq_rest: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub vq: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerDecoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderVectorQuantization<B: Backend> {
    pub _codebook: Qwen3TtsSpeechTokenizerDecoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderCodebook<B: Backend> {
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embedding_sum: Param<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub enum Qwen3TtsSpeechTokenizerWaveDecoderEntry<B: Backend> {
    InputConv(Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B>),
    UpsampleStage(Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage<B>),
    OutputActivation(TokenizerSnakeBeta<B>),
    OutputConv(Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage<B: Backend> {
    pub block: (
        TokenizerSnakeBeta<B>,
        TokenizerCausalTransConv1d<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B: Backend> {
    pub act1: TokenizerSnakeBeta<B>,
    pub conv1: TokenizerCausalConv1d<B>,
    pub act2: TokenizerSnakeBeta<B>,
    pub conv2: TokenizerCausalConv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderBackbone<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderBackboneLayer<B>>,
}

#[derive(Module, Debug)]
pub enum Qwen3TtsSpeechTokenizerEncoderBackboneLayer<B: Backend> {
    InputConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    Resnet(Qwen3TtsSpeechTokenizerEncoderResnetLayer<B>),
    DownsampleConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    OutputConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    Empty(Qwen3TtsSpeechTokenizerEmptyModule),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderConvLayer<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderResnetLayer<B: Backend> {
    pub block: (
        Qwen3TtsSpeechTokenizerEmptyModule,
        TokenizerCausalConv1d<B>,
        Qwen3TtsSpeechTokenizerEmptyModule,
        TokenizerCausalConv1d<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderTransformerLayer<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsSpeechTokenizerEncoderAttention<B>,
    pub mlp: Qwen3TtsSpeechTokenizerEncoderMlp<B>,
    pub input_layernorm: LayerNorm<B>,
    pub post_attention_layernorm: LayerNorm<B>,
    pub self_attn_layer_scale: TokenizerLayerScale<B>,
    pub mlp_layer_scale: TokenizerLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderQuantizer<B: Backend> {
    pub semantic_residual_vector_quantizer:
        Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B>,
    pub acoustic_residual_vector_quantizer:
        Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderVectorQuantization<B: Backend> {
    pub codebook: Qwen3TtsSpeechTokenizerEncoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderCodebook<B: Backend> {
    pub initialized: Param<Tensor<B, 1>>,
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embed_sum: Param<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct TokenizerCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct TokenizerCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

#[derive(Module, Debug)]
pub struct TokenizerSnakeBeta<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct TokenizerLayerScale<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug, Default, Clone)]
pub struct Qwen3TtsSpeechTokenizerEmptyModule;

#[derive(Debug)]
pub struct LoadedQwen3TtsSpeechTokenizer<B: Backend> {
    pub config: Qwen3TtsSpeechTokenizerConfig,
    pub model: Qwen3TtsSpeechTokenizerCheckpoint<B>,
    pub load_report: LoadReport,
    pub model_dir: PathBuf,
    pub weights_path: PathBuf,
}

impl Qwen3TtsSpeechTokenizerConfig {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = model_dir
            .as_ref()
            .join("speech_tokenizer")
            .join("config.json");
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }

    pub fn init_checkpoint<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsSpeechTokenizerCheckpoint<B> {
        Qwen3TtsSpeechTokenizerCheckpoint {
            decoder: self.decoder_config.init(device),
            encoder: self.init_encoder(device),
        }
    }

    fn init_encoder<B: Backend>(&self, device: &B::Device) -> Qwen3TtsSpeechTokenizerEncoder<B> {
        let frame_rate = self.input_sample_rate as f64 / self.encode_downsample_rate as f64;
        let backbone_frame_rate = self.encoder_config.sampling_rate as f64
            / self
                .encoder_config
                .upsampling_ratios
                .iter()
                .copied()
                .product::<usize>() as f64;
        debug_assert!((frame_rate - self.encoder_config._frame_rate).abs() < 1e-6);
        let downsample_factor =
            (backbone_frame_rate / self.encoder_config._frame_rate).round() as usize;
        let downsample_kernel = downsample_factor * 2;

        Qwen3TtsSpeechTokenizerEncoder {
            encoder: self.encoder_config.init_backbone::<B>(device),
            encoder_transformer: self.encoder_config.init_transformer::<B>(device),
            downsample: TokenizerCausalConv1d::<B>::new(
                self.encoder_config.hidden_size,
                self.encoder_config.hidden_size,
                downsample_kernel,
                2,
                1,
                1,
                false,
                device,
            ),
            quantizer: self.encoder_config.init_quantizer::<B>(device),
        }
    }
}

impl Qwen3TtsSpeechTokenizerDecoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TtsSpeechTokenizerDecoder<B> {
        let mut wave_decoder: Vec<Qwen3TtsSpeechTokenizerWaveDecoderEntry<B>> =
            Vec::with_capacity(self.upsample_rates.len() + 3);
        wave_decoder.push(Qwen3TtsSpeechTokenizerWaveDecoderEntry::InputConv(
            Qwen3TtsSpeechTokenizerWaveDecoderConvEntry {
                conv: Conv1dConfig::new(self.latent_dim, self.decoder_dim, 7)
                    .with_bias(true)
                    .init(device),
            },
        ));

        for layer_idx in 0..self.upsample_rates.len() {
            wave_decoder.push(self.init_wave_decoder_block(layer_idx, device));
        }

        let output_dim = self.decoder_dim / (1 << self.upsample_rates.len());
        wave_decoder.push(Qwen3TtsSpeechTokenizerWaveDecoderEntry::OutputActivation(
            TokenizerSnakeBeta::<B>::new(output_dim, device),
        ));
        wave_decoder.push(Qwen3TtsSpeechTokenizerWaveDecoderEntry::OutputConv(
            Qwen3TtsSpeechTokenizerWaveDecoderConvEntry {
                conv: Conv1dConfig::new(output_dim, 1, 7)
                    .with_bias(true)
                    .init(device),
            },
        ));

        Qwen3TtsSpeechTokenizerDecoder {
            pre_transformer: self.init_pre_transformer(device),
            quantizer: self.init_quantizer(device),
            pre_conv: TokenizerCausalConv1d::<B>::new(
                self.codebook_dim,
                self.latent_dim,
                3,
                1,
                1,
                1,
                true,
                device,
            ),
            upsample: self
                .upsampling_ratios
                .iter()
                .map(|&ratio| {
                    (
                        TokenizerCausalTransConv1d::<B>::new(
                            self.latent_dim,
                            self.latent_dim,
                            ratio,
                            ratio,
                            1,
                            true,
                            device,
                        ),
                        Qwen3TtsSpeechTokenizerConvNeXtBlock {
                            dwconv: TokenizerCausalConv1d::<B>::new(
                                self.latent_dim,
                                self.latent_dim,
                                7,
                                1,
                                1,
                                self.latent_dim,
                                true,
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
    ) -> Qwen3TtsSpeechTokenizerDecoderTransformer<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsSpeechTokenizerDecoderTransformer {
            layers: (0..self.num_hidden_layers)
                .map(|_| Qwen3TtsSpeechTokenizerDecoderTransformerLayer {
                    self_attn: Qwen3TtsSpeechTokenizerDecoderAttention {
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
                    mlp: Qwen3TtsSpeechTokenizerDecoderMlp {
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
                    self_attn_layer_scale: TokenizerLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                    mlp_layer_scale: TokenizerLayerScale::new(
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
    ) -> Qwen3TtsSpeechTokenizerDecoderQuantizer<B> {
        let hidden = self.codebook_dim / 2;

        Qwen3TtsSpeechTokenizerDecoderQuantizer {
            rvq_first: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer {
                input_proj: Conv1dConfig::new(self.codebook_dim, hidden, 1)
                    .with_bias(false)
                    .init(device),
                output_proj: Conv1dConfig::new(hidden, self.codebook_dim, 1)
                    .with_bias(false)
                    .init(device),
                vq: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization {
                    layers: vec![Qwen3TtsSpeechTokenizerDecoderVectorQuantization {
                        _codebook: Qwen3TtsSpeechTokenizerDecoderCodebook::new(
                            self.codebook_size,
                            hidden,
                            device,
                        ),
                    }],
                },
            },
            rvq_rest: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer {
                input_proj: Conv1dConfig::new(self.codebook_dim, hidden, 1)
                    .with_bias(false)
                    .init(device),
                output_proj: Conv1dConfig::new(hidden, self.codebook_dim, 1)
                    .with_bias(false)
                    .init(device),
                vq: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization {
                    layers: (0..self
                        .num_quantizers
                        .saturating_sub(self.num_semantic_quantizers))
                        .map(|_| Qwen3TtsSpeechTokenizerDecoderVectorQuantization {
                            _codebook: Qwen3TtsSpeechTokenizerDecoderCodebook::new(
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
    ) -> Qwen3TtsSpeechTokenizerWaveDecoderEntry<B> {
        let in_dim = self.decoder_dim / (1 << layer_idx);
        let out_dim = self.decoder_dim / (1 << (layer_idx + 1));
        let upsample_rate = self.upsample_rates[layer_idx];

        Qwen3TtsSpeechTokenizerWaveDecoderEntry::UpsampleStage(
            Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage {
                block: (
                    TokenizerSnakeBeta::<B>::new(in_dim, device),
                    TokenizerCausalTransConv1d::<B>::new(
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
    ) -> Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B> {
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit {
            act1: TokenizerSnakeBeta::<B>::new(channels, device),
            conv1: TokenizerCausalConv1d::<B>::new(
                channels, channels, 7, 1, dilation, 1, true, device,
            ),
            act2: TokenizerSnakeBeta::<B>::new(channels, device),
            conv2: TokenizerCausalConv1d::<B>::new(channels, channels, 1, 1, 1, 1, true, device),
        }
    }
}

impl Qwen3TtsSpeechTokenizerEncoderConfig {
    fn init_backbone<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsSpeechTokenizerEncoderBackbone<B> {
        let mut layers = std::iter::repeat_with(|| {
            Qwen3TtsSpeechTokenizerEncoderBackboneLayer::Empty(Qwen3TtsSpeechTokenizerEmptyModule)
        })
        .take(15)
        .collect::<Vec<Qwen3TtsSpeechTokenizerEncoderBackboneLayer<B>>>();

        layers[0] = Qwen3TtsSpeechTokenizerEncoderBackboneLayer::InputConv(
            Qwen3TtsSpeechTokenizerEncoderConvLayer {
                conv: Conv1dConfig::new(self.audio_channels, self.num_filters, self.kernel_size)
                    .with_bias(true)
                    .init(device),
            },
        );

        let mut scaling = 1;
        let mut residual_positions = [1usize, 4, 7, 10].into_iter();
        let mut conv_positions = [3usize, 6, 9, 12].into_iter();
        for ratio in self.upsampling_ratios.iter().rev().copied() {
            let current_scale = scaling * self.num_filters;

            let position = residual_positions.next().expect("fixed encoder layout");
            let hidden = current_scale / self.compress;
            layers[position] = Qwen3TtsSpeechTokenizerEncoderBackboneLayer::Resnet(
                Qwen3TtsSpeechTokenizerEncoderResnetLayer {
                    block: (
                        Qwen3TtsSpeechTokenizerEmptyModule,
                        TokenizerCausalConv1d::<B>::new(
                            current_scale,
                            hidden,
                            self.residual_kernel_size,
                            1,
                            1,
                            1,
                            true,
                            device,
                        ),
                        Qwen3TtsSpeechTokenizerEmptyModule,
                        TokenizerCausalConv1d::<B>::new(
                            hidden,
                            current_scale,
                            1,
                            1,
                            1,
                            1,
                            true,
                            device,
                        ),
                    ),
                },
            );

            let position = conv_positions.next().expect("fixed encoder layout");
            layers[position] = Qwen3TtsSpeechTokenizerEncoderBackboneLayer::DownsampleConv(
                Qwen3TtsSpeechTokenizerEncoderConvLayer {
                    conv: Conv1dConfig::new(current_scale, current_scale * 2, ratio * 2)
                        .with_stride(ratio)
                        .with_bias(true)
                        .init(device),
                },
            );
            scaling *= 2;
        }

        layers[14] = Qwen3TtsSpeechTokenizerEncoderBackboneLayer::OutputConv(
            Qwen3TtsSpeechTokenizerEncoderConvLayer {
                conv: Conv1dConfig::new(
                    scaling * self.num_filters,
                    self.hidden_size,
                    self.last_kernel_size,
                )
                .with_bias(true)
                .init(device),
            },
        );

        Qwen3TtsSpeechTokenizerEncoderBackbone { layers }
    }

    fn init_transformer<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsSpeechTokenizerEncoderTransformer<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsSpeechTokenizerEncoderTransformer {
            layers: (0..self.num_hidden_layers)
                .map(|_| Qwen3TtsSpeechTokenizerEncoderTransformerLayer {
                    self_attn: Qwen3TtsSpeechTokenizerEncoderAttention {
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
                    mlp: Qwen3TtsSpeechTokenizerEncoderMlp {
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
                    self_attn_layer_scale: TokenizerLayerScale::new(
                        self.hidden_size,
                        self.layer_scale_initial_scale,
                        device,
                    ),
                    mlp_layer_scale: TokenizerLayerScale::new(
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
    ) -> Qwen3TtsSpeechTokenizerEncoderQuantizer<B> {
        Qwen3TtsSpeechTokenizerEncoderQuantizer {
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
    ) -> Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B> {
        Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer {
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
                .map(|_| Qwen3TtsSpeechTokenizerEncoderVectorQuantization {
                    codebook: Qwen3TtsSpeechTokenizerEncoderCodebook::new(
                        self.codebook_size,
                        self.codebook_dim,
                        device,
                    ),
                })
                .collect(),
        }
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerDecoderCodebook<B> {
    fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embedding_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerEncoderCodebook<B> {
    fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            initialized: Initializer::Ones.init([1], device),
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embed_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}

impl<B: Backend> TokenizerCausalConv1d<B> {
    fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: Conv1dConfig::new(channels_in, channels_out, kernel_size)
                .with_stride(stride)
                .with_dilation(dilation)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
        }
    }
}

impl<B: Backend> TokenizerCausalTransConv1d<B> {
    fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: ConvTranspose1dConfig::new([channels_in, channels_out], kernel_size)
                .with_stride(stride)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
        }
    }
}

impl<B: Backend> TokenizerSnakeBeta<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            alpha: Initializer::Zeros.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }
}

impl<B: Backend> TokenizerLayerScale<B> {
    fn new(channels: usize, initial_scale: f64, device: &B::Device) -> Self {
        Self {
            scale: Initializer::Constant {
                value: initial_scale,
            }
            .init([channels], device),
        }
    }
}

pub fn load_qwen3_tts_speech_tokenizer<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsSpeechTokenizer<B>, Qwen3TtsLoadError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let weights_path = model_dir.join("speech_tokenizer").join("model.safetensors");
    let config = Qwen3TtsSpeechTokenizerConfig::load_from_model_dir(&model_dir)?;
    let mut model = config.init_checkpoint(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(speech_tokenizer_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    let load_report = LoadReport {
        applied: apply_result.applied.len(),
        skipped: apply_result.skipped.len(),
        missing: apply_result.missing.len(),
        unused: apply_result.unused.len(),
    };

    Ok(LoadedQwen3TtsSpeechTokenizer {
        config,
        model,
        load_report,
        model_dir,
        weights_path,
    })
}

pub fn verify_qwen3_tts_speech_tokenizer_weights<B: Backend>(
    model: &Qwen3TtsSpeechTokenizerCheckpoint<B>,
    weights_path: impl AsRef<Path>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    verify_module_weights(
        model,
        weights_path,
        Some(speech_tokenizer_export_key_remapper()),
        artifacts,
    )
}

fn speech_tokenizer_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![(
        r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.weight$",
        "${1}.gamma",
    )])
    .expect("static regex remapping must compile")
}

fn speech_tokenizer_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![(
        r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.gamma$",
        "${1}.weight",
    )])
    .expect("static regex remapping must compile")
}

#[cfg(test)]
mod tests {
    use crate::{VerificationArtifacts, default_workspace_root, find_local_qwen_tts_model_dir};

    use super::*;

    type TestBackend = burn::backend::Flex;

    #[test]
    fn real_checkpoint_speech_tokenizer_weights_roundtrip() {
        let workspace_root = default_workspace_root();
        let model_dir =
            find_local_qwen_tts_model_dir(&workspace_root).expect("local qwen model directory");
        let device = Default::default();

        let loaded = load_qwen3_tts_speech_tokenizer::<TestBackend>(&model_dir, &device)
            .expect("speech tokenizer checkpoint should load");
        assert_eq!(loaded.load_report.missing, 0);
        assert_eq!(loaded.load_report.skipped, 0);
        assert_eq!(loaded.load_report.applied, 496);

        let artifacts = VerificationArtifacts::new(
            workspace_root.join("artifacts/qwen3_tts/speech_tokenizer/test_roundtrip"),
        );
        let verification = verify_qwen3_tts_speech_tokenizer_weights(
            &loaded.model,
            &loaded.weights_path,
            Some(&artifacts),
        )
        .expect("speech tokenizer should roundtrip back to the source checkpoint");
        assert_eq!(verification.tensor_count, loaded.load_report.applied);
        assert_eq!(verification.tensor_count, 496);
    }
}
