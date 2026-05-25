use burn::module::Initializer;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{LayerNormConfig, LinearConfig, RmsNormConfig};
use burn::tensor::backend::Backend;

use crate::shared::config::tokenizer::Qwen3TtsSpeechTokenizerDecoderConfig;
use crate::speech_tokenizer::model::common::{
    TokenizerCausalConv1d, TokenizerCausalTransConv1d, TokenizerSnakeBeta,
};
use crate::speech_tokenizer::model::decoder::{
    Qwen3TtsSpeechTokenizerConvNeXtBlock, Qwen3TtsSpeechTokenizerDecoder,
    Qwen3TtsSpeechTokenizerDecoderAttention, Qwen3TtsSpeechTokenizerDecoderMlp,
    Qwen3TtsSpeechTokenizerDecoderQuantizer,
    Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization,
    Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer,
    Qwen3TtsSpeechTokenizerDecoderTransformer, Qwen3TtsSpeechTokenizerDecoderTransformerLayer,
    Qwen3TtsSpeechTokenizerDecoderVectorQuantization,
};
use crate::speech_tokenizer::model::wave_decoder::{
    Qwen3TtsSpeechTokenizerWaveDecoderConvEntry, Qwen3TtsSpeechTokenizerWaveDecoderEntry,
    Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit,
    Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage,
};
use crate::speech_tokenizer::{Qwen3TtsSpeechTokenizerDecoderCodebook, TokenizerLayerScale};

impl Qwen3TtsSpeechTokenizerDecoderConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TtsSpeechTokenizerDecoder<B> {
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

    pub(crate) fn init_quantizer<B: Backend>(
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
