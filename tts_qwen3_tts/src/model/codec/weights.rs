use std::path::Path;

use burn::module::Initializer;
use burn::nn::conv::Conv1dConfig;
use burn::nn::PaddingConfig1d;
use burn::nn::{LayerNormConfig, LinearConfig, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use super::activation::{AudioCodecLayerScale, AudioCodecSnakeBeta};
use super::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d};
use crate::model::codec::config::{
    Qwen3TtsAudioCodecConfig, Qwen3TtsAudioCodecDecoderConfig, Qwen3TtsAudioCodecEncoderConfig,
};
use crate::model::codec::core::{
    Qwen3TtsAudioCodec, Qwen3TtsAudioCodecConvNeXtBlock, Qwen3TtsAudioCodecDecoder,
    Qwen3TtsAudioCodecDecoderAttention, Qwen3TtsAudioCodecDecoderCodebook,
    Qwen3TtsAudioCodecDecoderMlp, Qwen3TtsAudioCodecDecoderQuantizer,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantization,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantizer, Qwen3TtsAudioCodecDecoderTransformer,
    Qwen3TtsAudioCodecDecoderTransformerLayer, Qwen3TtsAudioCodecDecoderVectorQuantization,
    Qwen3TtsAudioCodecWaveDecoderConvEntry, Qwen3TtsAudioCodecWaveDecoderEntry,
    Qwen3TtsAudioCodecWaveDecoderResidualUnit, Qwen3TtsAudioCodecWaveDecoderUpsampleStage,
};
use crate::model::codec::core::{
    Qwen3TtsAudioCodecEncoder, Qwen3TtsAudioCodecEncoderAttention,
    Qwen3TtsAudioCodecEncoderBackbone, Qwen3TtsAudioCodecEncoderBackboneLayer,
    Qwen3TtsAudioCodecEncoderCodebook, Qwen3TtsAudioCodecEncoderConvLayer,
    Qwen3TtsAudioCodecEncoderMlp, Qwen3TtsAudioCodecEncoderQuantizer,
    Qwen3TtsAudioCodecEncoderResidualVectorQuantizer, Qwen3TtsAudioCodecEncoderResnetLayer,
    Qwen3TtsAudioCodecEncoderTransformer, Qwen3TtsAudioCodecEncoderTransformerLayer,
    Qwen3TtsAudioCodecEncoderVectorQuantization,
};
use crate::Qwen3TtsLoadError;

const SPEECH_TOKENIZER_LOAD_KEY_PATTERNS: [(&str, &str); 1] = [(
    r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.weight$",
    "${1}.gamma",
)];
#[cfg(test)]
const SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS: [(&str, &str); 1] = [(
    r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.gamma$",
    "${1}.weight",
)];

fn audio_codec_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(SPEECH_TOKENIZER_LOAD_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

#[cfg(test)]
fn audio_codec_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

#[derive(Debug)]
pub struct LoadedQwen3TtsAudioCodec<B: Backend> {
    pub config: Qwen3TtsAudioCodecConfig,
    pub model: Qwen3TtsAudioCodec<B>,
}

pub fn load_qwen3_tts_audio_codec<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsAudioCodec<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts audio codec"
    );
    let config = Qwen3TtsAudioCodecConfig::load_from_path(&config_path)?;
    let mut model = config.init_model(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(audio_codec_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    tracing::info!(
        applied = apply_result.applied.len(),
        skipped = apply_result.skipped.len(),
        missing = apply_result.missing.len(),
        unused = apply_result.unused.len(),
        "loaded qwen3 tts audio codec weights"
    );

    Ok(LoadedQwen3TtsAudioCodec { config, model })
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

impl Qwen3TtsAudioCodecConfig {
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
                device,
            ),
            quantizer: self.encoder_config.init_quantizer::<B>(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{audio_codec_export_key_remapper, audio_codec_load_key_remapper};

    #[test]
    fn audio_codec_remappers_compile() {
        let _ = audio_codec_load_key_remapper();
        let _ = audio_codec_export_key_remapper();
    }
}

impl Qwen3TtsAudioCodecEncoderConfig {
    fn init_backbone<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsAudioCodecEncoderBackbone<B> {
        let mut layers = Vec::with_capacity(self.upsampling_ratios.len() * 2 + 3);
        layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(
            Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(self.audio_channels, self.num_filters, self.kernel_size)
                    .with_bias(true)
                    .init(device),
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
                        device,
                    ),
                },
            ));

            layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(
                Qwen3TtsAudioCodecEncoderConvLayer {
                    conv: Conv1dConfig::new(current_scale, current_scale * 2, ratio * 2)
                        .with_stride(ratio)
                        .with_bias(true)
                        .init(device),
                },
            ));
            scaling *= 2;
        }

        layers.push(Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(
            Qwen3TtsAudioCodecEncoderConvLayer {
                conv: Conv1dConfig::new(
                    scaling * self.num_filters,
                    self.hidden_size,
                    self.last_kernel_size,
                )
                .with_bias(true)
                .init(device),
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
                channels, channels, 7, 1, dilation, 1, true, device,
            ),
            act2: AudioCodecSnakeBeta::<B>::new(channels, device),
            conv2: AudioCodecCausalConv1d::<B>::new(channels, channels, 1, 1, 1, 1, true, device),
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
