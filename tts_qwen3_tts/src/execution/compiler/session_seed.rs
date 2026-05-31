use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

use crate::execution::compiler::SemanticRequestCondition;
use crate::model::talker::config::Qwen3TtsTalkerConfig;
use crate::model::talker::weights::LoadedQwen3TtsTalker;
use crate::Qwen3TtsInferenceError;

#[derive(Debug)]
pub(crate) struct SessionSeed<B: Backend> {
    pub(crate) inputs_embeds: Tensor<B, 3>,
    pub(crate) position_ids: Tensor<B, 3, Int>,
    pub(crate) attention_mask: Tensor<B, 2, Int>,
    pub(crate) trailing_text_hidden: Tensor<B, 3>,
    pub(crate) tts_pad_embed: Tensor<B, 3>,
    pub(crate) reference_codec_frames: Option<Vec<Vec<i64>>>,
    pub(crate) max_new_tokens: usize,
    pub(crate) codec_eos_token_id: usize,
    pub(crate) suppress_token_ids: Vec<usize>,
}

struct SeedEmbeddings<B: Backend> {
    inputs_embeds: Tensor<B, 3>,
    trailing_text_hidden: Tensor<B, 3>,
    tts_pad_embed: Tensor<B, 3>,
    reference_codec_frames: Option<Vec<Vec<i64>>>,
}

pub(crate) fn materialize_session_seed<B: Backend>(
    condition: &SemanticRequestCondition,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    device: &B::Device,
) -> Result<SessionSeed<B>, Qwen3TtsInferenceError> {
    let prepared = match condition.prompt_recipe {
        crate::execution::compiler::Qwen3TtsPromptRecipe::CustomVoiceInstructed => {
            let instruct_ids = condition.instruct_token_ids.as_deref().ok_or_else(|| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: "custom-voice instruct recipe requires instruct tokens".to_string(),
                }
            })?;
            build_non_streaming_seed(
                talker,
                &condition.text_token_ids,
                Some(instruct_ids),
                None,
                &condition.controls,
                talker_config.hidden_size,
                device,
            )
        }
        crate::execution::compiler::Qwen3TtsPromptRecipe::BaseVoiceCloneXVectorOnly => {
            let voice_clone = condition.voice_clone.as_ref().ok_or_else(|| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: "base voice-clone recipe requires compiled voice clone state"
                        .to_string(),
                }
            })?;
            build_voice_clone_seed(
                talker,
                &condition.text_token_ids,
                None,
                voice_clone,
                &condition.controls,
                talker_config.hidden_size,
                device,
            )
        }
        crate::execution::compiler::Qwen3TtsPromptRecipe::BaseVoiceCloneIcl => {
            let voice_clone = condition.voice_clone.as_ref().ok_or_else(|| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: "base voice-clone ICL recipe requires compiled voice clone state"
                        .to_string(),
                }
            })?;
            build_voice_clone_seed(
                talker,
                &condition.text_token_ids,
                Some(voice_clone.ref_text_token_ids.as_deref().ok_or_else(|| {
                    Qwen3TtsInferenceError::InvalidInput {
                        message: "base voice-clone ICL recipe requires reference text tokens"
                            .to_string(),
                    }
                })?),
                voice_clone,
                &condition.controls,
                talker_config.hidden_size,
                device,
            )
        }
        _ => build_non_streaming_seed(
            talker,
            &condition.text_token_ids,
            None,
            None,
            &condition.controls,
            talker_config.hidden_size,
            device,
        ),
    }?;

    let preferred_dtype = preferred_hidden_dtype::<B>(device);
    let seq_len = prepared.inputs_embeds.dims()[1];
    let inputs_embeds = prepared.inputs_embeds.cast(preferred_dtype);
    let attention_mask =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1; seq_len], [1, seq_len]), device);
    let position_data = (0..3)
        .flat_map(|_| (0..seq_len).map(|pos| pos as i32))
        .collect::<Vec<_>>();
    let position_ids =
        Tensor::<B, 3, Int>::from_data(TensorData::new(position_data, [3, 1, seq_len]), device);

    Ok(SessionSeed {
        inputs_embeds,
        position_ids,
        attention_mask,
        trailing_text_hidden: prepared.trailing_text_hidden.cast(preferred_dtype),
        tts_pad_embed: prepared.tts_pad_embed.cast(preferred_dtype),
        reference_codec_frames: prepared.reference_codec_frames,
        max_new_tokens: condition.max_new_tokens,
        codec_eos_token_id: condition.codec_eos_token_id,
        suppress_token_ids: build_suppress_token_ids(
            talker.config.vocab_size,
            condition.codec_eos_token_id,
        ),
    })
}

fn build_suppress_token_ids(vocab_size: usize, codec_eos_token_id: usize) -> Vec<usize> {
    (vocab_size.saturating_sub(1024)..vocab_size)
        .filter(|id| *id != codec_eos_token_id)
        .collect()
}

fn preferred_hidden_dtype<B: Backend>(device: &B::Device) -> DType {
    if B::supports_dtype(device, DType::BF16) {
        DType::BF16
    } else {
        DType::F32
    }
}

fn build_non_streaming_seed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    leading_text_ids: Option<&[i64]>,
    reference_codec_frames: Option<Vec<Vec<i64>>>,
    controls: &crate::execution::compiler::ProfileControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Result<SeedEmbeddings<B>, Qwen3TtsInferenceError> {
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

    let leading_embeds = leading_text_ids.map(|ids| project_text_ids(talker, ids, device));
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
            &std::iter::repeat_n(controls.codec_pad_id, body_len).collect::<Vec<_>>(),
            device,
        );
    let eos_with_codec_pad =
        tts_eos_embed + embed_codec_ids(talker, &[controls.codec_pad_id], device);
    let generation_bos =
        tts_pad_embed.clone() + embed_codec_ids(talker, &[controls.codec_bos_id], device);

    let mut segments = Vec::with_capacity(6);
    if let Some(leading_embeds) = leading_embeds {
        segments.push(leading_embeds);
    }
    segments.push(role_embeds);
    segments.push(prefix_embeds);
    segments.push(text_with_codec_pad);
    segments.push(eos_with_codec_pad);
    segments.push(generation_bos);
    Ok(SeedEmbeddings {
        inputs_embeds: Tensor::cat(segments, 1),
        trailing_text_hidden: tts_pad_embed.clone(),
        tts_pad_embed,
        reference_codec_frames,
    })
}

fn build_voice_clone_seed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    ref_text_token_ids: Option<&[i64]>,
    voice_clone: &crate::execution::compiler::CompiledVoiceCloneCondition,
    controls: &crate::execution::compiler::ProfileControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Result<SeedEmbeddings<B>, Qwen3TtsInferenceError> {
    let (tts_bos_embed, tts_eos_embed, tts_pad_embed) =
        special_text_embeds(talker, controls, hidden_size, device);
    let role_embeds = project_text_ids(talker, &text_ids[..3], device);
    let codec_prefix_embeds = voice_clone_codec_prefix(
        talker,
        voice_clone,
        controls,
        hidden_size,
        &tts_bos_embed,
        &tts_pad_embed,
        device,
    )?;

    if let Some(ref_text_token_ids) = ref_text_token_ids {
        let ref_codec_frames = voice_clone.ref_codec_token_ids.clone().ok_or_else(|| {
            Qwen3TtsInferenceError::InvalidInput {
                message: "voice-clone ICL requires reference codec frames".to_string(),
            }
        })?;
        let target_body_ids = &text_ids[3..text_ids.len() - 5];
        let ref_body_ids = &ref_text_token_ids[3..ref_text_token_ids.len() - 2];
        let mut all_text_ids = Vec::with_capacity(ref_body_ids.len() + target_body_ids.len() + 1);
        all_text_ids.extend_from_slice(ref_body_ids);
        all_text_ids.extend_from_slice(target_body_ids);
        all_text_ids.push(controls.tts_eos_token_id);

        let text_embed = project_text_ids(talker, &all_text_ids, device);
        let text_len = text_embed.dims()[1];

        let ref_codec_embeds =
            sum_ref_codec_embeddings(talker, &ref_codec_frames, hidden_size, device)?;
        let codec_bos_embed = embed_codec_ids(talker, &[controls.codec_bos_id], device);
        let codec_embed = Tensor::cat(vec![codec_bos_embed, ref_codec_embeds], 1);
        let codec_len = codec_embed.dims()[1];

        let (icl_embeds, trailing_text_hidden) = if text_len > codec_len {
            let text_head = text_embed
                .clone()
                .slice([0..1, 0..codec_len, 0..hidden_size]);
            let trailing = text_embed
                .clone()
                .slice([0..1, codec_len..text_len, 0..hidden_size]);
            (text_head + codec_embed, trailing)
        } else {
            let padded_text = if codec_len > text_len {
                Tensor::cat(
                    vec![
                        text_embed,
                        tts_pad_embed.clone().repeat_dim(1, codec_len - text_len),
                    ],
                    1,
                )
            } else {
                text_embed
            };
            (padded_text + codec_embed, tts_pad_embed.clone())
        };

        Ok(SeedEmbeddings {
            inputs_embeds: Tensor::cat(vec![role_embeds, codec_prefix_embeds, icl_embeds], 1),
            trailing_text_hidden,
            tts_pad_embed,
            reference_codec_frames: Some(ref_codec_frames),
        })
    } else {
        let body_ids = &text_ids[3..text_ids.len() - 5];
        let mut prefill_segments = vec![role_embeds, codec_prefix_embeds];
        if let Some(first_text_with_codec_bos) =
            build_first_text_with_codec_bos(talker, body_ids, controls, device)
        {
            prefill_segments.push(first_text_with_codec_bos);
        }

        Ok(SeedEmbeddings {
            inputs_embeds: Tensor::cat(prefill_segments, 1),
            trailing_text_hidden: build_trailing_text_hidden(
                talker,
                &body_ids.get(1..).unwrap_or(&[]),
                &tts_eos_embed,
                device,
            ),
            tts_pad_embed,
            reference_codec_frames: None,
        })
    }
}

fn build_first_text_with_codec_bos<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    body_ids: &[i64],
    controls: &crate::execution::compiler::ProfileControlIds,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    let first_id = *body_ids.first()?;
    let first_text_embed = project_text_ids(talker, &[first_id], device);
    let codec_bos_embed = embed_codec_ids(talker, &[controls.codec_bos_id], device);
    Some(first_text_embed + codec_bos_embed)
}

fn build_trailing_text_hidden<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    remaining_body_ids: &[i64],
    tts_eos_embed: &Tensor<B, 3>,
    device: &B::Device,
) -> Tensor<B, 3> {
    if remaining_body_ids.is_empty() {
        return tts_eos_embed.clone();
    }

    Tensor::cat(
        vec![
            project_text_ids(talker, remaining_body_ids, device),
            tts_eos_embed.clone(),
        ],
        1,
    )
}

fn special_text_embeds<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    controls: &crate::execution::compiler::ProfileControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
    let special_embeds = project_text_ids(
        talker,
        &[
            controls.tts_bos_token_id,
            controls.tts_eos_token_id,
            controls.tts_pad_token_id,
        ],
        device,
    );
    (
        special_embeds.clone().slice([0..1, 0..1, 0..hidden_size]),
        special_embeds.clone().slice([0..1, 1..2, 0..hidden_size]),
        special_embeds.slice([0..1, 2..3, 0..hidden_size]),
    )
}

fn voice_clone_codec_prefix<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    voice_clone: &crate::execution::compiler::CompiledVoiceCloneCondition,
    controls: &crate::execution::compiler::ProfileControlIds,
    hidden_size: usize,
    tts_bos_embed: &Tensor<B, 3>,
    tts_pad_embed: &Tensor<B, 3>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    let prefix_tag_ids = &controls.codec_prefix_ids[..controls.codec_prefix_ids.len() - 2];
    let mut codec_segments = vec![embed_codec_ids(talker, prefix_tag_ids, device)];
    codec_segments.push(speaker_embedding_tensor(
        &voice_clone.speaker_embedding,
        hidden_size,
        tts_bos_embed.dtype(),
        device,
    )?);
    codec_segments.push(embed_codec_ids(talker, &[controls.codec_pad_id], device));
    let codec_prefix_embeds = Tensor::cat(codec_segments, 1);
    let codec_prefix_len = codec_prefix_embeds.dims()[1];
    let prefix_text_embeds = Tensor::cat(
        vec![
            tts_pad_embed
                .clone()
                .repeat_dim(1, codec_prefix_len.saturating_sub(1)),
            tts_bos_embed.clone(),
        ],
        1,
    );
    Ok(prefix_text_embeds + codec_prefix_embeds)
}

fn speaker_embedding_tensor<B: Backend>(
    embedding: &[f32],
    hidden_size: usize,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    if embedding.len() != hidden_size {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "speaker embedding hidden size mismatch: expected {hidden_size}, got {}",
                embedding.len()
            ),
        });
    }
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(embedding.to_vec(), [1, 1, hidden_size]),
        device,
    )
    .cast(dtype))
}

fn sum_ref_codec_embeddings<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    frames: &[Vec<i64>],
    hidden_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    if frames.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "reference codec frame list must be non-empty".to_string(),
        });
    }
    let num_groups = talker.config.num_code_groups;
    if frames.iter().any(|frame| frame.len() != num_groups) {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("reference codec frames must all contain {num_groups} groups"),
        });
    }

    let mut summed = embed_codec_group_batch(
        talker,
        frames
            .iter()
            .map(|frame| frame[0])
            .collect::<Vec<_>>()
            .as_slice(),
        0,
        device,
    );
    for group_idx in 1..num_groups {
        let group_embed = embed_codec_group_batch(
            talker,
            frames
                .iter()
                .map(|frame| frame[group_idx])
                .collect::<Vec<_>>()
                .as_slice(),
            group_idx,
            device,
        );
        summed = summed + group_embed;
    }
    let [batch, seq, actual_hidden] = summed.dims();
    if batch != 1 || actual_hidden != hidden_size {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "reference codec embedding shape mismatch: expected [1, T, {hidden_size}], got [{batch}, {seq}, {actual_hidden}]"
            ),
        });
    }
    Ok(summed)
}

fn embed_codec_group_batch<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    group_idx: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            [1, ids.len()],
        ),
        device,
    );
    if group_idx == 0 {
        talker.model.talker.model.codec_embedding.forward(tensor)
    } else {
        talker.model.talker.code_predictor.model.codec_embedding[group_idx - 1].forward(tensor)
    }
}

fn project_text_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            [1, ids.len()],
        ),
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
        TensorData::new(
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            [1, ids.len()],
        ),
        device,
    );
    talker.model.talker.model.codec_embedding.forward(tensor)
}
