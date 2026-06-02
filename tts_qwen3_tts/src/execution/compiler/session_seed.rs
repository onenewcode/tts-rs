use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use crate::Qwen3TtsInferenceError;
use crate::execution::audio_finalize::reference_codec_prefix_tensor;
use crate::execution::compiler::{
    CompiledVoiceCloneCondition, CompiledVoiceCloneConditionSource, ProfileControlIds,
    Qwen3TtsPromptRecipe, SemanticRequestCondition,
};
use crate::execution::reference_audio::load_reference_audio;
use crate::model::codec::weights::LoadedQwen3TtsAudioCodec;
use crate::model::speaker::LoadedQwen3TtsSpeakerEncoder;
use crate::model::talker::config::Qwen3TtsTalkerConfig;
use crate::model::talker::weights::LoadedQwen3TtsTalker;

#[derive(Debug)]
pub(crate) struct SessionSeed<B: Backend> {
    pub(crate) inputs_embeds: Tensor<B, 3>,
    pub(crate) position_ids: Tensor<B, 3, Int>,
    pub(crate) attention_mask: Tensor<B, 2, Int>,
    pub(crate) trailing_text_hidden: Tensor<B, 3>,
    pub(crate) tts_pad_embed: Tensor<B, 3>,
    pub(crate) reference_codec_prefix: Option<Tensor<B, 3, Int>>,
    pub(crate) reference_codec_frame_count: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) codec_eos_token_id: i64,
    pub(crate) sampling: crate::SamplingConfig,
    pub(crate) suppress_token_ids: Vec<usize>,
}

struct PreparedSeed<B: Backend> {
    inputs_embeds: Tensor<B, 3>,
    trailing_text_hidden: Tensor<B, 3>,
    tts_pad_embed: Tensor<B, 3>,
    reference_codec_prefix: Option<Tensor<B, 3, Int>>,
    reference_codec_frame_count: usize,
}

struct VoiceCloneState<B: Backend> {
    speaker_embedding: Tensor<B, 3>,
    reference_codec_prefix: Option<Tensor<B, 3, Int>>,
    reference_codec_frame_count: usize,
}

pub(crate) fn materialize_session_seed<B: Backend>(
    condition: &SemanticRequestCondition,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    decoder: &LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: Option<&LoadedQwen3TtsSpeakerEncoder<B>>,
    device: &B::Device,
) -> Result<SessionSeed<B>, Qwen3TtsInferenceError>
where
    B::Device: Clone,
{
    let prepared = match condition.prompt_recipe {
        Qwen3TtsPromptRecipe::CustomVoiceInstructed => build_non_streaming_seed(
            talker,
            &condition.text_token_ids,
            Some(instruct_ids(condition)?),
            &condition.controls,
            device,
        ),
        Qwen3TtsPromptRecipe::BaseVoiceCloneIcl
        | Qwen3TtsPromptRecipe::BaseVoiceCloneXVectorOnly => build_voice_clone_seed_from_condition(
            condition,
            talker,
            decoder,
            speaker_encoder,
            talker_config.hidden_size,
            device,
        ),
        _ => build_non_streaming_seed(
            talker,
            &condition.text_token_ids,
            None,
            &condition.controls,
            device,
        ),
    }?;

    let seq_len = prepared.inputs_embeds.dims()[1];
    let attention_mask = Tensor::<B, 2, Int>::ones([1, seq_len], device);
    let seq_len_i64 = i64::try_from(seq_len).map_err(|_| Qwen3TtsInferenceError::InvalidInput {
        message: format!("prompt sequence length {seq_len} does not fit the model int tensor"),
    })?;
    let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len_i64, device)
        .reshape([1, 1, seq_len])
        .repeat_dim(0, 3);

    Ok(SessionSeed {
        inputs_embeds: prepared.inputs_embeds,
        position_ids,
        attention_mask,
        trailing_text_hidden: prepared.trailing_text_hidden,
        tts_pad_embed: prepared.tts_pad_embed,
        reference_codec_prefix: prepared.reference_codec_prefix,
        reference_codec_frame_count: prepared.reference_codec_frame_count,
        max_new_tokens: condition.max_new_tokens,
        codec_eos_token_id: condition.codec_eos_token_id,
        sampling: condition.sampling.clone(),
        suppress_token_ids: build_suppress_token_ids(
            talker.config.vocab_size,
            condition.codec_eos_token_id,
        ),
    })
}

fn instruct_ids(condition: &SemanticRequestCondition) -> Result<&[i64], Qwen3TtsInferenceError> {
    condition
        .instruct_token_ids
        .as_deref()
        .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
            message: "custom-voice instruct recipe requires instruct tokens".to_string(),
        })
}

fn voice_clone_condition(
    condition: &SemanticRequestCondition,
) -> Result<&CompiledVoiceCloneCondition, Qwen3TtsInferenceError> {
    condition
        .voice_clone
        .as_ref()
        .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
            message: "base voice-clone recipe requires compiled voice clone state".to_string(),
        })
}

fn reference_text_ids(
    condition: &SemanticRequestCondition,
) -> Result<&[i64], Qwen3TtsInferenceError> {
    voice_clone_condition(condition)?
        .ref_text_token_ids
        .as_deref()
        .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
            message: "base voice-clone ICL recipe requires reference text tokens".to_string(),
        })
}

fn build_suppress_token_ids(vocab_size: usize, codec_eos_token_id: i64) -> Vec<usize> {
    let eos_token_id = usize::try_from(codec_eos_token_id).ok();
    (vocab_size.saturating_sub(1024)..vocab_size)
        .filter(|id| Some(*id) != eos_token_id)
        .collect()
}

fn build_non_streaming_seed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    leading_text_ids: Option<&[i64]>,
    controls: &ProfileControlIds,
    device: &B::Device,
) -> Result<PreparedSeed<B>, Qwen3TtsInferenceError> {
    let (tts_bos_embed, tts_eos_embed, tts_pad_embed) =
        special_text_embeds(talker, controls, device);
    let role_embeds = project_text_ids(talker, &text_ids[..3], device);
    let body_embeds = project_text_ids(talker, &text_ids[3..text_ids.len() - 5], device);
    let leading_embeds = leading_text_ids.map(|ids| project_text_ids(talker, ids, device));

    let codec_len = controls.codec_prefix_ids.len();
    let codec_prefix_embeds =
        embed_codec_ids(talker, &controls.codec_prefix_ids[..codec_len - 1], device);
    let prefix_embeds = Tensor::cat(
        vec![
            tts_pad_embed
                .clone()
                .repeat_dim(1, codec_len.saturating_sub(2)),
            tts_bos_embed,
        ],
        1,
    ) + codec_prefix_embeds;

    let body_len = body_embeds.dims()[1];
    let codec_embedding = &talker.model.talker.model.codec_embedding;
    let text_with_codec_pad = body_embeds
        + codec_embedding.forward(Tensor::<B, 2, Int>::full(
            [1, body_len],
            controls.codec_pad_id,
            device,
        ));
    let eos_with_codec_pad = tts_eos_embed
        + codec_embedding.forward(Tensor::<B, 2, Int>::full(
            [1, 1],
            controls.codec_pad_id,
            device,
        ));
    let generation_bos = tts_pad_embed.clone()
        + codec_embedding.forward(Tensor::<B, 2, Int>::full(
            [1, 1],
            controls.codec_bos_id,
            device,
        ));

    let mut segments = Vec::with_capacity(6);
    if let Some(leading_embeds) = leading_embeds {
        segments.push(leading_embeds);
    }
    segments.push(role_embeds);
    segments.push(prefix_embeds);
    segments.push(text_with_codec_pad);
    segments.push(eos_with_codec_pad);
    segments.push(generation_bos);

    Ok(PreparedSeed {
        inputs_embeds: Tensor::cat(segments, 1),
        trailing_text_hidden: tts_pad_embed.clone(),
        tts_pad_embed,
        reference_codec_prefix: None,
        reference_codec_frame_count: 0,
    })
}

fn build_voice_clone_seed_from_condition<B: Backend>(
    condition: &SemanticRequestCondition,
    talker: &LoadedQwen3TtsTalker<B>,
    decoder: &LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: Option<&LoadedQwen3TtsSpeakerEncoder<B>>,
    hidden_size: usize,
    device: &B::Device,
) -> Result<PreparedSeed<B>, Qwen3TtsInferenceError>
where
    B::Device: Clone,
{
    let voice_clone = materialize_voice_clone_condition(
        voice_clone_condition(condition)?,
        talker,
        decoder,
        speaker_encoder,
        hidden_size,
        device,
    )?;
    let ref_text_ids = matches!(
        condition.prompt_recipe,
        Qwen3TtsPromptRecipe::BaseVoiceCloneIcl
    )
    .then(|| reference_text_ids(condition))
    .transpose()?;

    build_voice_clone_seed(
        talker,
        &condition.text_token_ids,
        ref_text_ids,
        voice_clone,
        &condition.controls,
        hidden_size,
        device,
    )
}

fn build_voice_clone_seed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    ref_text_ids: Option<&[i64]>,
    voice_clone: VoiceCloneState<B>,
    controls: &ProfileControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Result<PreparedSeed<B>, Qwen3TtsInferenceError> {
    let VoiceCloneState {
        speaker_embedding,
        reference_codec_prefix,
        reference_codec_frame_count,
    } = voice_clone;
    let (tts_bos_embed, tts_eos_embed, tts_pad_embed) =
        special_text_embeds(talker, controls, device);
    let role_embeds = project_text_ids(talker, &text_ids[..3], device);
    let codec_prefix = voice_clone_codec_prefix(
        talker,
        speaker_embedding,
        controls,
        hidden_size,
        tts_bos_embed,
        &tts_pad_embed,
        device,
    )?;

    match ref_text_ids {
        Some(ref_text_ids) => build_voice_clone_icl_seed(
            talker,
            text_ids,
            ref_text_ids,
            role_embeds,
            codec_prefix,
            reference_codec_prefix.ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                message: "voice-clone ICL requires reference codec frames".to_string(),
            })?,
            reference_codec_frame_count,
            tts_pad_embed,
            controls,
            hidden_size,
            device,
        ),
        None => {
            let body_ids = &text_ids[3..text_ids.len() - 5];
            let mut prefill = vec![role_embeds, codec_prefix];
            if let Some(first_text_with_codec_bos) =
                build_first_text_with_codec_bos(talker, body_ids, controls, device)
            {
                prefill.push(first_text_with_codec_bos);
            }

            Ok(PreparedSeed {
                inputs_embeds: Tensor::cat(prefill, 1),
                trailing_text_hidden: build_trailing_text_hidden(
                    talker,
                    body_ids.get(1..).unwrap_or(&[]),
                    tts_eos_embed,
                    device,
                ),
                tts_pad_embed,
                reference_codec_prefix: None,
                reference_codec_frame_count: 0,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_voice_clone_icl_seed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    ref_text_ids: &[i64],
    role_embeds: Tensor<B, 3>,
    codec_prefix: Tensor<B, 3>,
    reference_codec_prefix: Tensor<B, 3, Int>,
    reference_codec_frame_count: usize,
    tts_pad_embed: Tensor<B, 3>,
    controls: &ProfileControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Result<PreparedSeed<B>, Qwen3TtsInferenceError> {
    let target_body_ids = &text_ids[3..text_ids.len() - 5];
    let reference_body_ids = &ref_text_ids[3..ref_text_ids.len() - 2];
    let mut all_text_ids = Vec::with_capacity(reference_body_ids.len() + target_body_ids.len() + 1);
    all_text_ids.extend_from_slice(reference_body_ids);
    all_text_ids.extend_from_slice(target_body_ids);
    all_text_ids.push(controls.tts_eos_token_id);

    let text_embeds = project_text_ids(talker, &all_text_ids, device);
    let text_len = text_embeds.dims()[1];
    let reference_codec_embeds =
        sum_ref_codec_embeddings(talker, reference_codec_prefix.clone(), hidden_size)?;
    let codec_embeds = Tensor::cat(
        vec![
            talker
                .model
                .talker
                .model
                .codec_embedding
                .forward(Tensor::<B, 2, Int>::full(
                    [1, 1],
                    controls.codec_bos_id,
                    device,
                )),
            reference_codec_embeds,
        ],
        1,
    );
    let codec_len = codec_embeds.dims()[1];

    let (icl_embeds, trailing_text_hidden) = if text_len > codec_len {
        let text_head = text_embeds
            .clone()
            .slice([0..1, 0..codec_len, 0..hidden_size]);
        let trailing = text_embeds.slice([0..1, codec_len..text_len, 0..hidden_size]);
        (text_head + codec_embeds, trailing)
    } else {
        let padded_text = if codec_len > text_len {
            Tensor::cat(
                vec![
                    text_embeds,
                    tts_pad_embed.clone().repeat_dim(1, codec_len - text_len),
                ],
                1,
            )
        } else {
            text_embeds
        };
        (padded_text + codec_embeds, tts_pad_embed.clone())
    };

    Ok(PreparedSeed {
        inputs_embeds: Tensor::cat(vec![role_embeds, codec_prefix, icl_embeds], 1),
        trailing_text_hidden,
        tts_pad_embed,
        reference_codec_prefix: Some(reference_codec_prefix),
        reference_codec_frame_count,
    })
}

fn materialize_voice_clone_condition<B: Backend>(
    voice_clone: &CompiledVoiceCloneCondition,
    talker: &LoadedQwen3TtsTalker<B>,
    decoder: &LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: Option<&LoadedQwen3TtsSpeakerEncoder<B>>,
    hidden_size: usize,
    device: &B::Device,
) -> Result<VoiceCloneState<B>, Qwen3TtsInferenceError>
where
    B::Device: Clone,
{
    let hidden_dtype = talker_hidden_dtype(talker);
    match &voice_clone.source {
        CompiledVoiceCloneConditionSource::Prompt(prompt) => Ok(VoiceCloneState {
            speaker_embedding: speaker_embedding_tensor(
                &prompt.speaker_embedding,
                hidden_size,
                hidden_dtype,
                device,
            )?,
            reference_codec_prefix: prompt
                .ref_codec_token_ids
                .as_ref()
                .map(|frames| {
                    reference_codec_prefix_tensor::<B>(
                        frames,
                        1,
                        talker.config.num_code_groups,
                        device,
                    )
                })
                .transpose()?,
            reference_codec_frame_count: prompt.ref_codec_token_ids.as_ref().map_or(0, Vec::len),
        }),
        CompiledVoiceCloneConditionSource::ReferenceAudio(reference) => {
            let speaker_encoder =
                speaker_encoder.ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                    message:
                        "voice clone prompt requires a Base model with speaker_encoder weights"
                            .to_string(),
                })?;
            let speaker_audio =
                load_reference_audio(&reference.path, speaker_encoder.sample_rate())?;
            let speaker_embedding = speaker_encoder
                .encode_embedding(&speaker_audio.samples)
                .reshape([1, 1, hidden_size]);
            let speaker_embedding = if speaker_embedding.dtype() == hidden_dtype {
                speaker_embedding
            } else {
                speaker_embedding.cast(hidden_dtype)
            };
            let (reference_codec_prefix, reference_codec_frame_count) = if reference.x_vector_only {
                (None, 0)
            } else {
                let codec_audio = load_reference_audio(
                    &reference.path,
                    u32::try_from(decoder.config.input_sample_rate).map_err(|_| {
                        Qwen3TtsInferenceError::InvalidInput {
                            message: format!(
                                "audio codec reference input sample rate {} exceeds the supported u32 audio range",
                                decoder.config.input_sample_rate
                            ),
                        }
                    })?,
                )?;
                let reference_codec_prefix =
                    decoder.encode_reference_codec_prefix(device, &codec_audio.samples)?;
                let reference_codec_frame_count = reference_codec_prefix.dims()[2];
                (Some(reference_codec_prefix), reference_codec_frame_count)
            };
            Ok(VoiceCloneState {
                speaker_embedding,
                reference_codec_prefix,
                reference_codec_frame_count,
            })
        }
    }
}

fn build_first_text_with_codec_bos<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    body_ids: &[i64],
    controls: &ProfileControlIds,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    let first_id = *body_ids.first()?;
    Some(
        project_text_ids(talker, &[first_id], device)
            + talker
                .model
                .talker
                .model
                .codec_embedding
                .forward(Tensor::<B, 2, Int>::full(
                    [1, 1],
                    controls.codec_bos_id,
                    device,
                )),
    )
}

fn build_trailing_text_hidden<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    remaining_body_ids: &[i64],
    tts_eos_embed: Tensor<B, 3>,
    device: &B::Device,
) -> Tensor<B, 3> {
    if remaining_body_ids.is_empty() {
        return tts_eos_embed;
    }

    Tensor::cat(
        vec![
            project_text_ids(talker, remaining_body_ids, device),
            tts_eos_embed,
        ],
        1,
    )
}

fn special_text_embeds<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    controls: &ProfileControlIds,
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
    let hidden_size = special_embeds.dims()[2];
    (
        special_embeds.clone().slice([0..1, 0..1, 0..hidden_size]),
        special_embeds.clone().slice([0..1, 1..2, 0..hidden_size]),
        special_embeds.slice([0..1, 2..3, 0..hidden_size]),
    )
}

fn voice_clone_codec_prefix<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    speaker_embedding: Tensor<B, 3>,
    controls: &ProfileControlIds,
    hidden_size: usize,
    tts_bos_embed: Tensor<B, 3>,
    tts_pad_embed: &Tensor<B, 3>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    let [batch_size, steps, speaker_hidden] = speaker_embedding.dims();
    if [batch_size, steps, speaker_hidden] != [1, 1, hidden_size] {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "speaker embedding shape mismatch: expected [1, 1, {hidden_size}], got [{batch_size}, {steps}, {speaker_hidden}]"
            ),
        });
    }

    let prefix_tag_ids = &controls.codec_prefix_ids[..controls.codec_prefix_ids.len() - 2];
    let prefix_codec = embed_codec_ids(talker, prefix_tag_ids, device);
    let codec_prefix = Tensor::cat(
        vec![
            prefix_codec,
            speaker_embedding,
            talker
                .model
                .talker
                .model
                .codec_embedding
                .forward(Tensor::<B, 2, Int>::full(
                    [1, 1],
                    controls.codec_pad_id,
                    device,
                )),
        ],
        1,
    );
    let prefix_len = codec_prefix.dims()[1];
    let prefix_text = Tensor::cat(
        vec![
            tts_pad_embed
                .clone()
                .repeat_dim(1, prefix_len.saturating_sub(1)),
            tts_bos_embed,
        ],
        1,
    );

    Ok(prefix_text + codec_prefix)
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

    let embedding =
        Tensor::<B, 1>::from_data(embedding, (device, dtype)).reshape([1, 1, hidden_size]);
    Ok(embedding)
}

fn talker_hidden_dtype<B: Backend>(talker: &LoadedQwen3TtsTalker<B>) -> DType {
    talker
        .model
        .talker
        .model
        .codec_embedding
        .weight
        .val()
        .dtype()
}

fn sum_ref_codec_embeddings<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    reference_codec_prefix: Tensor<B, 3, Int>,
    hidden_size: usize,
) -> Result<Tensor<B, 3>, Qwen3TtsInferenceError> {
    let [batch_size, num_groups, time_steps] = reference_codec_prefix.dims();
    if batch_size != 1 || time_steps == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "reference codec prefix must have shape [1, groups, time>0], got [{batch_size}, {num_groups}, {time_steps}]"
            ),
        });
    }
    if num_groups != talker.config.num_code_groups {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "reference codec prefix uses {num_groups} groups, expected {}",
                talker.config.num_code_groups
            ),
        });
    }

    let mut groups = reference_codec_prefix.chunk(num_groups, 1).into_iter();
    let mut summed = embed_codec_group_slice(
        talker,
        groups
            .next()
            .expect("reference codec prefix should include the first group"),
        0,
    );
    for (group_idx, group_tokens) in groups.enumerate() {
        summed = summed + embed_codec_group_slice(talker, group_tokens, group_idx + 1);
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

fn embed_codec_group_slice<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    token_ids: Tensor<B, 3, Int>,
    group_idx: usize,
) -> Tensor<B, 3> {
    let time_steps = token_ids.dims()[2];
    let token_ids = token_ids.reshape([1, time_steps]);
    if group_idx == 0 {
        talker.model.talker.model.codec_embedding.forward(token_ids)
    } else {
        talker.model.talker.code_predictor.model.codec_embedding[group_idx - 1].forward(token_ids)
    }
}

fn project_text_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 1, Int>::from_ints(ids, device).reshape([1, ids.len()]);
    talker
        .model
        .talker
        .text_projection
        .forward(talker.model.talker.model.text_embedding.forward(tensor))
}

fn embed_codec_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 1, Int>::from_ints(ids, device).reshape([1, ids.len()]);
    talker.model.talker.model.codec_embedding.forward(tensor)
}
