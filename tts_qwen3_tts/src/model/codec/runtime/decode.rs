use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::model::codec::config::Qwen3TtsAudioCodecDecoderConfig;
use crate::model::codec::loading::LoadedQwen3TtsAudioCodec;

#[derive(Debug, Clone)]
pub struct Waveform {
    sample_rate: u32,
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
        if !samples.len().is_multiple_of(batch_size * channels) {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform element mismatch: {} samples do not fit batch={batch_size}, channels={channels}",
                    samples.len()
                ),
            });
        }
        Ok(Self {
            sample_rate,
            samples,
        })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn samples(&self) -> &[f32] {
        &self.samples
    }
}

pub fn decode_waveform<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    codec_ids: Tensor<B, 3, Int>,
) -> Result<Tensor<B, 3>, QwenTtsInferenceError> {
    validate_codec_input_3d(&codec_ids, &loaded.config.decoder_config)?;

    let config = &loaded.config.decoder_config;
    let rope = RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim)
        .with_theta(config.rope_theta as f32)
        .init(&codec_ids.device());

    Ok(loaded.model.decoder.forward(
        codec_ids,
        config.num_semantic_quantizers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        &rope,
    ))
}

pub fn lift_waveform<B: Backend>(
    sample_rate: u32,
    waveform: Tensor<B, 3>,
) -> Result<Waveform, QwenTtsInferenceError> {
    let [batch_size, channels, _time_steps] = waveform.dims();
    let samples = waveform
        .try_into_data()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("failed to read waveform: {source}"),
        })?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("failed to read waveform: {source}"),
        })?;
    Waveform::new(sample_rate, batch_size, channels, samples)
}

pub fn waveform_to_pcm(waveform: &Waveform) -> Result<Vec<i16>, QwenTtsInferenceError> {
    Ok(waveform
        .samples()
        .iter()
        .copied()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect())
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
