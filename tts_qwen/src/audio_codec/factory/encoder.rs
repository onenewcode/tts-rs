use burn::nn::conv::Conv1dConfig;
use burn::nn::{LayerNormConfig, LinearConfig};
use burn::tensor::backend::Backend;

use crate::audio_codec::model::decoder::Qwen3TtsAudioCodecCheckpoint;
use crate::audio_codec::model::encoder::{
    Qwen3TtsAudioCodecEncoder, Qwen3TtsAudioCodecEncoderAttention,
    Qwen3TtsAudioCodecEncoderBackbone, Qwen3TtsAudioCodecEncoderBackboneLayer,
    Qwen3TtsAudioCodecEncoderCodebook, Qwen3TtsAudioCodecEncoderConvLayer,
    Qwen3TtsAudioCodecEncoderMlp, Qwen3TtsAudioCodecEncoderQuantizer,
    Qwen3TtsAudioCodecEncoderResidualVectorQuantizer, Qwen3TtsAudioCodecEncoderResnetLayer,
    Qwen3TtsAudioCodecEncoderTransformer, Qwen3TtsAudioCodecEncoderTransformerLayer,
    Qwen3TtsAudioCodecEncoderVectorQuantization,
};
use crate::shared::config::audio_codec::{
    Qwen3TtsAudioCodecConfig, Qwen3TtsAudioCodecEncoderConfig,
};
use crate::shared::nn::activation::{AudioCodecLayerScale, Qwen3TtsAudioCodecEmptyModule};
use crate::shared::nn::conv::AudioCodecCausalConv1d;

const ENCODER_BACKBONE_LEN: usize = 15;
const ENCODER_RESIDUAL_POSITIONS: [usize; 4] = [1, 4, 7, 10];
const ENCODER_DOWNSAMPLE_POSITIONS: [usize; 4] = [3, 6, 9, 12];

pub(crate) fn derive_encoder_downsample_factor(
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

pub(crate) fn derive_encoder_downsample_kernel(downsample_factor: usize) -> usize {
    downsample_factor * 2
}

impl Qwen3TtsAudioCodecConfig {
    pub fn init_checkpoint<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecCheckpoint<B> {
        Qwen3TtsAudioCodecCheckpoint {
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
                device,
            ),
            quantizer: self.encoder_config.init_quantizer::<B>(device),
        }
    }
}

impl Qwen3TtsAudioCodecEncoderConfig {
    pub(crate) fn init_backbone<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderBackbone<B> {
        let mut layers = std::iter::repeat_with(|| {
            Qwen3TtsAudioCodecEncoderBackboneLayer::Empty(Qwen3TtsAudioCodecEmptyModule)
        })
        .take(ENCODER_BACKBONE_LEN)
        .collect::<Vec<Qwen3TtsAudioCodecEncoderBackboneLayer<B>>>();

        layers[0] =
            Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(self.audio_channels, self.num_filters, self.kernel_size)
                    .with_bias(true)
                    .init(device),
            });

        let mut scaling = 1;
        let mut residual_positions = ENCODER_RESIDUAL_POSITIONS.into_iter();
        let mut conv_positions = ENCODER_DOWNSAMPLE_POSITIONS.into_iter();
        for ratio in self.upsampling_ratios.iter().rev().copied() {
            let current_scale = scaling * self.num_filters;

            let position = residual_positions.next().expect("fixed encoder layout");
            let hidden = current_scale / self.compress;
            layers[position] = Qwen3TtsAudioCodecEncoderBackboneLayer::Resnet(
                Qwen3TtsAudioCodecEncoderResnetLayer {
                    block: (
                        Qwen3TtsAudioCodecEmptyModule,
                        AudioCodecCausalConv1d::<B>::new(
                            current_scale,
                            hidden,
                            self.residual_kernel_size,
                            1,
                            1,
                            1,
                            true,
                            device,
                        ),
                        Qwen3TtsAudioCodecEmptyModule,
                        AudioCodecCausalConv1d::<B>::new(
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
            layers[position] = Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(
                Qwen3TtsAudioCodecEncoderConvLayer {
                    conv: Conv1dConfig::new(current_scale, current_scale * 2, ratio * 2)
                        .with_stride(ratio)
                        .with_bias(true)
                        .init(device),
                },
            );
            scaling *= 2;
        }

        layers[14] = Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(
            Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(
                    scaling * self.num_filters,
                    self.hidden_size,
                    self.last_kernel_size,
                )
                .with_bias(true)
                .init(device),
            },
        );

        Qwen3TtsAudioCodecEncoderBackbone { layers }
    }

    pub(crate) fn init_transformer<B: Backend>(
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

    pub(crate) fn init_quantizer<B: Backend>(
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
    pub(crate) fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        use burn::module::Initializer;
        Self {
            initialized: Initializer::Ones.init([1], device),
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embed_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}
