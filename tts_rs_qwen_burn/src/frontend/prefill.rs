use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

use crate::shared::io::LoadedQwen3TtsTalker;
use crate::{Qwen3TtsInferenceError, Qwen3TtsTalkerConfig};

use super::prompt::{build_custom_voice_prompt, resolve_custom_voice_control_ids};
use super::text_tokenizer::Qwen3TtsTextTokenizer;
use super::types::{CustomVoiceBatch, FrontendOutput};

pub fn build_custom_voice_prefill_batch<B: Backend>(
    tokenizer: &Qwen3TtsTextTokenizer,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    batch: &CustomVoiceBatch,
    device: &B::Device,
) -> Result<FrontendOutput<B>, Qwen3TtsInferenceError> {
    if batch.requests.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "custom voice batch must contain at least one request".to_string(),
        });
    }

    let mut text_token_ids = Vec::with_capacity(batch.requests.len());
    let mut codec_prefix_ids = Vec::with_capacity(batch.requests.len());
    let mut sample_embeddings = Vec::with_capacity(batch.requests.len());
    let mut seq_lens = Vec::with_capacity(batch.requests.len());

    for request in &batch.requests {
        let prompt = build_custom_voice_prompt(request);
        let text_ids = tokenizer.encode(&prompt).map_err(|source| {
            Qwen3TtsInferenceError::InvalidInput {
                message: format!("failed to tokenize custom voice prompt: {source}"),
            }
        })?;
        if text_ids.len() < 8 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "custom voice prompt tokenization is too short: {} tokens",
                    text_ids.len()
                ),
            });
        }
        let controls = resolve_custom_voice_control_ids(tokenizer.model_dir(), request)?;
        let prefix_ids = controls.codec_prefix_ids.clone();
        let sample = build_non_streaming_custom_voice_sample(
            talker,
            &text_ids,
            &controls,
            talker_config.hidden_size,
            device,
        );
        seq_lens.push(sample.dims()[1]);
        sample_embeddings.push(sample);
        text_token_ids.push(text_ids);
        codec_prefix_ids.push(prefix_ids);
    }

    let batch_size = batch.requests.len();
    let max_len = seq_lens.iter().copied().max().unwrap_or(0);
    let dtype = sample_embeddings[0].dtype();
    let mut padded_embeddings = Vec::with_capacity(batch_size);
    let mut attention_data = Vec::with_capacity(batch_size * max_len);
    let mut position_data = Vec::with_capacity(3 * batch_size * max_len);

    for (sample, seq_len) in sample_embeddings.into_iter().zip(seq_lens.iter().copied()) {
        let pad_len = max_len - seq_len;
        if pad_len > 0 {
            let pad = Tensor::<B, 3>::zeros([1, pad_len, talker_config.hidden_size], device)
                .cast(dtype);
            padded_embeddings.push(Tensor::cat(vec![pad, sample], 1));
        } else {
            padded_embeddings.push(sample);
        }
        attention_data.extend(std::iter::repeat(0).take(pad_len));
        attention_data.extend(std::iter::repeat(1).take(seq_len));
    }

    for axis in 0..3 {
        let _ = axis;
        for seq_len in seq_lens.iter().copied() {
            let pad_len = max_len - seq_len;
            position_data.extend(std::iter::repeat(0).take(pad_len));
            position_data.extend((0..seq_len).map(|pos| pos as i32));
        }
    }

    let inputs_embeds = Tensor::cat(padded_embeddings, 0).cast(DType::BF16);
    let attention_mask = Tensor::<B, 2, Int>::from_data(
        TensorData::new(attention_data, [batch_size, max_len]),
        device,
    );
    let position_ids = Tensor::<B, 3, Int>::from_data(
        TensorData::new(position_data, [3, batch_size, max_len]),
        device,
    );

    Ok(FrontendOutput {
        text_token_ids,
        codec_prefix_ids,
        inputs_embeds,
        position_ids,
        attention_mask,
    })
}

fn build_non_streaming_custom_voice_sample<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    controls: &super::prompt::CustomVoiceControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let special_embeds = project_text_ids(
        talker,
        &[
            controls.tts_bos_token_id,
            controls.tts_eos_token_id,
            controls.tts_pad_token_id,
        ],
        device,
    );
    let tts_bos_embed = special_embeds.clone().slice([0..1, 0..1, 0..hidden_size]);
    let tts_eos_embed = special_embeds.clone().slice([0..1, 1..2, 0..hidden_size]);
    let tts_pad_embed = special_embeds.slice([0..1, 2..3, 0..hidden_size]);

    let role_embeds = project_text_ids(talker, &text_ids[..3], device);
    let body_embeds = project_text_ids(talker, &text_ids[3..text_ids.len() - 5], device);

    let codec_embeds = embed_codec_ids(talker, &controls.codec_prefix_ids, device);
    let codec_len = controls.codec_prefix_ids.len();
    let codec_prefix_embeds = codec_embeds
        .clone()
        .slice([0..1, 0..codec_len - 1, 0..hidden_size]);
    let prefix_text_embeds = Tensor::cat(
        vec![
            tts_pad_embed
                .clone()
                .repeat_dim(1, codec_len.saturating_sub(2)),
            tts_bos_embed,
        ],
        1,
    );
    let prefix_embeds = prefix_text_embeds + codec_prefix_embeds;

    let body_len = body_embeds.dims()[1];
    let text_with_codec_pad = body_embeds
        + embed_codec_ids(
            talker,
            &std::iter::repeat(controls.codec_pad_id)
                .take(body_len)
                .collect::<Vec<_>>(),
            device,
        );
    let eos_with_codec_pad =
        tts_eos_embed + embed_codec_ids(talker, &[controls.codec_pad_id], device);
    let generation_bos =
        tts_pad_embed + embed_codec_ids(talker, &[controls.codec_bos_id], device);

    Tensor::cat(
        vec![
            role_embeds,
            prefix_embeds,
            text_with_codec_pad,
            eos_with_codec_pad,
            generation_bos,
        ],
        1,
    )
}

fn project_text_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|id| *id as i32).collect::<Vec<_>>(), [1, ids.len()]),
        device,
    );
    let embeds = talker.model.talker.model.text_embedding.forward(tensor);
    talker.model.talker.text_projection.forward(embeds)
}

fn embed_codec_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.iter().map(|id| *id as i32).collect::<Vec<_>>(), [1, ids.len()]),
        device,
    );
    talker.model.talker.model.codec_embedding.forward(tensor)
}
