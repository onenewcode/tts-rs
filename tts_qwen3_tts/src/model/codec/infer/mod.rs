use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::Qwen3TtsInferenceError;
use crate::error::QwenTtsInferenceError;
use crate::model::codec::config::Qwen3TtsAudioCodecDecoderConfig;
use crate::model::codec::weights::LoadedQwen3TtsAudioCodec;
use crate::model::nn::tensor::read_float_tensor_vec;

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

    pub(crate) fn from_tensor<B: Backend>(
        sample_rate: u32,
        waveform: Tensor<B, 3>,
    ) -> Result<Self, QwenTtsInferenceError> {
        let [batch_size, channels, _time_steps] = waveform.dims();
        let samples = read_float_tensor_vec(waveform, "failed to read waveform")?;
        Self::new(sample_rate, batch_size, channels, samples)
    }

    pub(crate) fn to_pcm(&self) -> Vec<i16> {
        self.samples()
            .iter()
            .copied()
            .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }
}

impl<B: Backend> LoadedQwen3TtsAudioCodec<B> {
    pub fn decode_waveform(
        &self,
        codec_ids: Tensor<B, 3, Int>,
    ) -> Result<Tensor<B, 3>, QwenTtsInferenceError> {
        validate_codec_input_3d(&codec_ids, &self.config.decoder_config)?;

        let config = &self.config.decoder_config;
        let rope = RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim)
            .with_theta(config.rope_theta as f32)
            .init(&codec_ids.device());

        Ok(self.model.decoder.forward(
            codec_ids,
            config.num_semantic_quantizers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            &rope,
        ))
    }

    pub fn encode_reference_codec_frames(
        &self,
        device: &B::Device,
        samples: &[f32],
    ) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
        let waveform = Tensor::<B, 3>::from_data(
            TensorData::new(samples.to_vec(), [1, 1, samples.len()]),
            device,
        );
        self.model.encoder.encode_reference_frames(
            &self.config.encoder_config,
            self.config.encoder_valid_num_quantizers,
            waveform,
        )
    }
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
