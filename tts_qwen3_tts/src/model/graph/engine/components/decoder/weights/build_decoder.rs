use burn::module::Initializer;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{LayerNormConfig, LinearConfig, RmsNormConfig};
use burn::tensor::backend::Backend;

use crate::kernels::activation::{AudioCodecLayerScale, AudioCodecSnakeBeta};
use crate::kernels::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d};
use crate::model::graph::engine::components::decoder::graph::audio_codec::decoder::{
    Qwen3TtsAudioCodecConvNeXtBlock, Qwen3TtsAudioCodecDecoder, Qwen3TtsAudioCodecDecoderAttention,
    Qwen3TtsAudioCodecDecoderCodebook, Qwen3TtsAudioCodecDecoderMlp,
    Qwen3TtsAudioCodecDecoderQuantizer, Qwen3TtsAudioCodecDecoderResidualVectorQuantization,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantizer, Qwen3TtsAudioCodecDecoderTransformer,
    Qwen3TtsAudioCodecDecoderTransformerLayer, Qwen3TtsAudioCodecDecoderVectorQuantization,
};
use crate::model::graph::engine::components::decoder::graph::audio_codec::wave_decoder::{
    Qwen3TtsAudioCodecWaveDecoderConvEntry, Qwen3TtsAudioCodecWaveDecoderEntry,
    Qwen3TtsAudioCodecWaveDecoderResidualUnit, Qwen3TtsAudioCodecWaveDecoderUpsampleStage,
};
use crate::model::graph::engine::components::decoder::import::config::Qwen3TtsAudioCodecDecoderConfig;
/// TODO 整个构建太过繁琐，需要进行优化
impl Qwen3TtsAudioCodecDecoderConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TtsAudioCodecDecoder<B> {
        let mut wave_decoder: Vec<Qwen3TtsAudioCodecWaveDecoderEntry<B>> =
            Vec::with_capacity(self.upsample_rates.len() + 3);
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

    pub(crate) fn init_quantizer<B: Backend>(
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
                channels, channels, 7, 1, dilation, 1, true, device,
            ),
            act2: AudioCodecSnakeBeta::<B>::new(channels, device),
            conv2: AudioCodecCausalConv1d::<B>::new(channels, channels, 1, 1, 1, 1, true, device),
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecDecoderCodebook<B> {
    pub(crate) fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embedding_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}
