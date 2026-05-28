use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::model::graph::engine::components::decoder::import::config::Qwen3TtsAudioCodecDecoderConfig;
use crate::model::graph::engine::components::decoder::weights::LoadedQwen3TtsAudioCodec;
use crate::error::QwenTtsInferenceError;
use crate::profiling::record_operator;

#[derive(Debug, Clone)]
pub struct Waveform {
    sample_rate: u32,
    batch_size: usize,
    channels: usize,
    samples: Vec<f32>,
}

impl Waveform {
    pub(crate) fn new(
        sample_rate: u32,
        batch_size: usize,
        channels: usize,
        samples: Vec<f32>,
    ) -> Result<Self, QwenTtsInferenceError> {
        if sample_rate == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "waveform sample rate must be non-zero".to_string(),
            });
        }
        if batch_size == 0 || channels == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform batch/channels must be non-zero, got batch={batch_size}, channels={channels}"
                ),
            });
        }
        if samples.is_empty() {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "waveform sample payload must be non-empty".to_string(),
            });
        }
        if samples.len() % (batch_size * channels) != 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform element mismatch: {} samples do not fit batch={batch_size}, channels={channels}",
                    samples.len()
                ),
            });
        }
        Ok(Self {
            sample_rate,
            batch_size,
            channels,
            samples,
        })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn samples(&self) -> &[f32] {
        &self.samples
    }
}

pub fn decode_waveform<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    codec_ids: Tensor<B, 3, Int>,
) -> Result<Tensor<B, 3>, QwenTtsInferenceError> {
    infer(loaded, codec_ids, &loaded.config.decoder_config)
}

pub fn waveform_to_pcm(waveform: &Waveform) -> Result<Vec<i16>, QwenTtsInferenceError> {
    let _frame_count = waveform.samples().len() / (waveform.batch_size() * waveform.channels());
    Ok(waveform
        .samples()
        .iter()
        .copied()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect())
}

fn infer<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    codec_ids: Tensor<B, 3, Int>,
    config: &Qwen3TtsAudioCodecDecoderConfig,
) -> Result<Tensor<B, 3>, QwenTtsInferenceError> {
    validate_codec_input_3d(&codec_ids, config)?;

    let rope_cfg = RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim)
        .with_theta(config.rope_theta as f32);
    let rope = rope_cfg.init(&codec_ids.device());

    let waveform = record_operator("codec.decode", || {
        loaded.model.decoder.forward(
            codec_ids,
            config.num_semantic_quantizers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            &rope,
        )
    });

    Ok(waveform)
}

fn validate_codec_input_3d<B: Backend>(
    codec_ids: &Tensor<B, 3, Int>,
    config: &Qwen3TtsAudioCodecDecoderConfig,
) -> Result<(), QwenTtsInferenceError> {
    let [batch, num_quantizers, _time_steps] = codec_ids.dims();
    if batch == 0 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: "codec batch size must be non-zero".to_string(),
        });
    }
    if num_quantizers != config.num_quantizers {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "codec token count {} doesn't match expected {} quantizer layers",
                num_quantizers, config.num_quantizers
            ),
        });
    }
    Ok(())
}
