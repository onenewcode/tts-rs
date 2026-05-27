use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::model::config::audio_codec::Qwen3TtsAudioCodecDecoderConfig;
use crate::model::load::audio_codec::LoadedQwen3TtsAudioCodec;
use crate::profiling::record_operator;

pub fn decode_waveform<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    codec_ids: Tensor<B, 3, Int>,
) -> Result<Tensor<B, 3>, QwenTtsInferenceError> {
    infer(loaded, codec_ids, &loaded.config.decoder_config)
}

pub fn waveform_to_pcm<B: Backend>(
    waveform: &Tensor<B, 3>,
) -> Result<Vec<i16>, QwenTtsInferenceError> {
    let samples: Vec<f32> = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| QwenTtsInferenceError::TensorRead {
            message: format!("failed to read waveform: {e}"),
        })?;
    Ok(samples
        .into_iter()
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
