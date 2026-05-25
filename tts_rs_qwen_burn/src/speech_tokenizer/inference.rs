use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::Qwen3TtsInferenceError;

use crate::shared::config::tokenizer::Qwen3TtsSpeechTokenizerDecoderConfig;
use super::load::LoadedQwen3TtsSpeechTokenizer;

/// Decode codec token IDs to audio waveform.
///
/// `codec_ids`: [batch, num_quantizers, time_steps] — one token per quantizer layer
///   for each time step. Each column along dim 1 is a different VQ layer's token.
///
/// Returns audio waveform: [batch, 1, num_samples].
pub fn decode_codec_tokens<B: Backend>(
    loaded: &LoadedQwen3TtsSpeechTokenizer<B>,
    codec_ids: Tensor<B, 3, Int>,
    config: &Qwen3TtsSpeechTokenizerDecoderConfig,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    validate_codec_input_3d(&codec_ids, config)?;

    let rope_cfg = RotaryEncodingConfig::new(
        config.max_position_embeddings,
        config.head_dim, // RoPE acts on per-head dim, not full Q/K dim
    )
    .with_theta(config.rope_theta as f32);
    let rope = rope_cfg.init(&codec_ids.device());

    let (waveform, _) = loaded.model.decoder.forward(
        codec_ids,
        config.num_semantic_quantizers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        &rope,
        false,
    );

    Ok(waveform)
}

/// Single-time-step convenience: `codec_ids` as `[batch, num_quantizers]`.
pub fn decode_codec_tokens_single_step<B: Backend>(
    loaded: &LoadedQwen3TtsSpeechTokenizer<B>,
    codec_ids: Tensor<B, 2, Int>,
    config: &Qwen3TtsSpeechTokenizerDecoderConfig,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    let [batch, num_q] = codec_ids.dims();
    let codec_3d = codec_ids.reshape([batch, num_q, 1]);
    decode_codec_tokens(loaded, codec_3d, config)
}

fn validate_codec_input_3d<B: Backend>(
    codec_ids: &Tensor<B, 3, Int>,
    config: &Qwen3TtsSpeechTokenizerDecoderConfig,
) -> Result<(), Qwen3TtsInferenceError> {
    let [batch, num_quantizers, _time_steps] = codec_ids.dims();
    if batch == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "codec batch size must be non-zero".to_string(),
        });
    }
    if num_quantizers != config.num_quantizers {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "codec token count {} doesn't match expected {} quantizer layers",
                num_quantizers, config.num_quantizers
            ),
        });
    }
    Ok(())
}
