//! Talker inference internals.
//!
//! The talker stage owns one entry point: `infer`, which performs prefill,
//! autoregressive talker token generation, and per-step code predictor expansion.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use std::time::Instant;

use super::model::build_attention_mask;
use super::nn::mlp::native_linear_3d;

use crate::Qwen3TtsInferenceError;
use crate::shared::config::talker::Qwen3TtsTalkerConfig;
use crate::shared::io::talker_load::LoadedQwen3TtsTalker;
use crate::shared::runtime::cache::KeyValueCache;
use crate::shared::runtime::sampling::{SamplingConfig, apply_repetition_penalty, sample_token};

#[derive(Debug)]
pub(crate) struct TalkerInferInput<B: Backend> {
    pub prefill_inputs_embeds: Tensor<B, 3>,
    pub prefill_position_ids: Tensor<B, 3, Int>,
    pub prefill_attention_mask: Option<Tensor<B, 2, Int>>,
    pub trailing_text_hidden: Option<Tensor<B, 3>>,
    pub tts_pad_embed: Option<Tensor<B, 3>>,
    pub sampling: SamplingConfig,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<usize>,
    pub suppress_token_ids: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct TalkerInferOutput<B: Backend> {
    pub talker_token_ids: Tensor<B, 2, Int>,
    pub codec_token_ids: Tensor<B, 3, Int>,
    pub generated_audio_steps: usize,
}

struct TalkerStepOutput<B: Backend> {
    last_hidden_state: Tensor<B, 3>,
    logits: Tensor<B, 3>,
}

pub(crate) fn infer<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerInferInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerInferOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    let started = Instant::now();
    validate_cache_layer_count(config, cache)?;
    if input.max_new_tokens == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "talker infer max_new_tokens must be greater than zero".to_string(),
        });
    }

    for layer_cache in cache.iter_mut() {
        layer_cache.reset();
    }

    let device = input.prefill_inputs_embeds.device();
    let [batch_size, prefill_len, hidden_size] = input.prefill_inputs_embeds.dims();
    tracing::info!(
        batch_size,
        prefill_len,
        hidden_size,
        max_new_tokens = input.max_new_tokens,
        eos_token_id = ?input.eos_token_id,
        suppress_token_count = input.suppress_token_ids.len(),
        "starting talker infer"
    );
    validate_generation_side_inputs(
        batch_size,
        config.hidden_size,
        input.trailing_text_hidden.as_ref().map(Tensor::dims),
        input.tts_pad_embed.as_ref().map(Tensor::dims),
    )?;

    let prefill_started = Instant::now();
    let prefill = prefill(
        config,
        loaded,
        input.prefill_inputs_embeds,
        input.prefill_position_ids,
        input.prefill_attention_mask,
        cache,
    )?;
    tracing::info!(
        elapsed_ms = prefill_started.elapsed().as_millis(),
        cache_len = validate_cache_lengths(cache)?,
        last_hidden_shape = ?prefill.last_hidden_state.dims(),
        logits_shape = ?prefill.logits.dims(),
        "finished talker prefill"
    );

    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits = apply_repetition_penalty(
        prefill.logits.clone(),
        &empty_history,
        input.sampling.repetition_penalty,
    );
    let (mut selected_token, mut eos_mask) = sample_token::<B>(
        prefill_logits,
        &input.sampling,
        input.eos_token_id,
        &input.suppress_token_ids,
        &device,
    );
    let mut talker_tokens = vec![selected_token.clone()];
    let mut current_hidden = last_hidden_step(prefill.last_hidden_state);
    let mut codec_steps = Vec::new();

    for step_idx in 0..input.max_new_tokens {
        if input.eos_token_id.is_some() && batch_finished(&eos_mask) {
            tracing::info!(step_idx, "stopping talker infer after EOS");
            break;
        }

        tracing::debug!(
            step_idx,
            talker_cache_len = validate_cache_lengths(cache)?,
            "starting talker decode step"
        );
        let mut predictor_cache = (0..config.code_predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    batch_size,
                    config.code_predictor_config.num_key_value_heads,
                    config.num_code_groups + 1,
                    config.code_predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();
        let predictor_started = Instant::now();
        let codec_ids = infer_code_predictor_groups(
            config,
            loaded,
            current_hidden.clone(),
            selected_token.clone(),
            &input.sampling,
            &mut predictor_cache,
        )?;
        tracing::debug!(
            step_idx,
            codec_group_shape = ?codec_ids.dims(),
            predictor_cache_len = validate_cache_lengths(&predictor_cache)?,
            elapsed_ms = predictor_started.elapsed().as_millis(),
            "finished code predictor expansion"
        );
        codec_steps.push(
            codec_ids
                .clone()
                .reshape([batch_size, config.num_code_groups, 1]),
        );

        if step_idx + 1 == input.max_new_tokens {
            tracing::info!(
                generated_steps = codec_steps.len(),
                "reached max_new_tokens in talker infer"
            );
            break;
        }

        let cache_len = validate_cache_lengths(cache)?;
        let inputs_embeds = add_trailing_text_embed(
            codec_group_context_embedding(config, loaded, codec_ids),
            input.trailing_text_hidden.as_ref(),
            input.tts_pad_embed.as_ref(),
            step_idx,
        );
        let position_ids = Tensor::<B, 3, Int>::full([3, batch_size, 1], cache_len as i32, &device);
        let decode_started = Instant::now();
        let decoded = decode_step(config, loaded, inputs_embeds, position_ids, cache)?;
        tracing::debug!(
            step_idx,
            elapsed_ms = decode_started.elapsed().as_millis(),
            cache_len_after = validate_cache_lengths(cache)?,
            logits_shape = ?decoded.logits.dims(),
            "finished talker decode step"
        );
        let past_ids = Tensor::cat(talker_tokens.clone(), 1);
        let penalized_logits =
            apply_repetition_penalty(decoded.logits, &past_ids, input.sampling.repetition_penalty);
        let (next_token, next_eos) = sample_token::<B>(
            penalized_logits,
            &input.sampling,
            input.eos_token_id,
            &input.suppress_token_ids,
            &device,
        );
        selected_token = next_token;
        eos_mask = eos_mask.bool_or(next_eos);
        talker_tokens.push(selected_token.clone());
        current_hidden = last_hidden_step(decoded.last_hidden_state);
    }

    if codec_steps.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "talker emitted EOS before any audio codec token".to_string(),
        });
    }

    let generated_audio_steps = codec_steps.len();
    let talker_token_ids = Tensor::cat(talker_tokens, 1);
    let codec_token_ids = Tensor::cat(codec_steps, 2);
    tracing::info!(
        talker_token_shape = ?talker_token_ids.dims(),
        codec_token_shape = ?codec_token_ids.dims(),
        generated_audio_steps,
        elapsed_ms = started.elapsed().as_millis(),
        "finished talker infer"
    );

    Ok(TalkerInferOutput {
        talker_token_ids,
        codec_token_ids,
        generated_audio_steps,
    })
}

fn prefill<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    inputs_embeds: Tensor<B, 3>,
    position_ids: Tensor<B, 3, Int>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerStepOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    tracing::debug!(
        input_shape = ?inputs_embeds.dims(),
        position_shape = ?position_ids.dims(),
        attention_mask_shape = ?attention_mask.as_ref().map(Tensor::dims),
        "running talker prefill"
    );
    validate_talker_input(
        "talker infer prefill",
        inputs_embeds.dims(),
        position_ids.dims(),
        attention_mask.as_ref().map(Tensor::dims),
        None,
    )?;

    let (last_hidden_state, logits) = loaded.model.talker.infer(
        inputs_embeds,
        position_ids,
        attention_mask,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        cache,
    );

    Ok(TalkerStepOutput {
        last_hidden_state,
        logits,
    })
}

fn decode_step<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    inputs_embeds: Tensor<B, 3>,
    position_ids: Tensor<B, 3, Int>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerStepOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    let cache_len = validate_cache_lengths(cache)?;
    tracing::debug!(
        input_shape = ?inputs_embeds.dims(),
        position_shape = ?position_ids.dims(),
        cache_len,
        "running talker decode"
    );
    validate_talker_input(
        "talker infer decode",
        inputs_embeds.dims(),
        position_ids.dims(),
        None,
        Some(cache_len),
    )?;

    let (last_hidden_state, logits) = loaded.model.talker.infer(
        inputs_embeds,
        position_ids,
        None,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        cache,
    );

    Ok(TalkerStepOutput {
        last_hidden_state,
        logits,
    })
}

fn infer_code_predictor_groups<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    talker_hidden_state: Tensor<B, 2>,
    base_codec_token_id: Tensor<B, 2, Int>,
    sampling: &SamplingConfig,
    cache: &mut [KeyValueCache<B>],
) -> Result<Tensor<B, 2, Int>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    let predictor_config = &config.code_predictor_config;
    validate_code_predictor_cache_layer_count(config, cache)?;
    if config.num_code_groups < 2 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "code predictor generation requires at least two code groups".to_string(),
        });
    }

    for layer_cache in cache.iter_mut() {
        layer_cache.reset();
    }

    let [batch_size, hidden_size] = talker_hidden_state.dims();
    tracing::debug!(
        batch_size,
        hidden_size,
        base_codec_shape = ?base_codec_token_id.dims(),
        "starting code predictor expansion"
    );
    let base_token_dims = base_codec_token_id.dims();
    if base_token_dims != [batch_size, 1] {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "code predictor base token shape mismatch: expected {:?}, got {:?}",
                [batch_size, 1],
                base_token_dims
            ),
        });
    }
    if hidden_size != config.hidden_size {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "code predictor hidden size mismatch: expected {}, got {}",
                config.hidden_size, hidden_size
            ),
        });
    }

    let base_embedding = loaded
        .model
        .talker
        .model
        .codec_embedding
        .forward(base_codec_token_id.clone());
    let prefill_inputs = Tensor::cat(
        vec![talker_hidden_state.unsqueeze::<3>(), base_embedding],
        1,
    );
    let prefill_hidden = run_code_predictor_hidden(config, loaded, prefill_inputs, None, cache);
    let prefill_head = &loaded.model.talker.code_predictor.lm_head[0];
    let prefill_logits = native_linear_3d(
        prefill_head,
        prefill_hidden
            .clone()
            .cast(prefill_head.weight.val().dtype()),
    );
    let device = prefill_logits.device();
    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits =
        apply_repetition_penalty(prefill_logits, &empty_history, sampling.repetition_penalty);
    let mut selected_token = sample_token::<B>(prefill_logits, sampling, None, &[], &device).0;
    let mut predictor_tokens = vec![selected_token.clone()];

    for head_idx in 1..config.num_code_groups - 1 {
        let step_inputs = loaded.model.talker.code_predictor.model.codec_embedding[head_idx - 1]
            .forward(selected_token);
        let step_hidden = run_code_predictor_hidden(config, loaded, step_inputs, None, cache);
        let head = &loaded.model.talker.code_predictor.lm_head[head_idx];
        let logits = native_linear_3d(head, step_hidden.clone().cast(head.weight.val().dtype()));
        let past_ids = Tensor::cat(predictor_tokens.clone(), 1);
        let logits = apply_repetition_penalty(logits, &past_ids, sampling.repetition_penalty);
        selected_token = sample_token::<B>(logits, sampling, None, &[], &device).0;
        predictor_tokens.push(selected_token.clone());
        tracing::debug!(
            head_idx,
            generated_tokens = predictor_tokens.len(),
            "sampled code predictor group"
        );
    }

    let predictor_token_ids = Tensor::cat(predictor_tokens, 1);
    let codec_ids = Tensor::cat(vec![base_codec_token_id, predictor_token_ids], 1);
    let final_cache_len = validate_cache_lengths(cache)?;
    if final_cache_len != config.num_code_groups {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "code predictor final cache length mismatch: expected {}, got {}",
                config.num_code_groups, final_cache_len
            ),
        });
    }

    debug_assert_eq!(
        predictor_config.num_code_groups, config.num_code_groups,
        "talker and code predictor code group counts should match"
    );
    tracing::debug!(
        final_cache_len,
        output_shape = ?codec_ids.dims(),
        "finished code predictor expansion"
    );
    Ok(codec_ids)
}

fn run_code_predictor_hidden<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    inputs_embeds: Tensor<B, 3>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    cache: &mut [KeyValueCache<B>],
) -> Tensor<B, 3>
where
    B: Backend,
{
    let predictor = &loaded.model.talker.code_predictor;
    let projected_inputs = if let Some(projection) = &predictor.small_to_mtp_projection {
        native_linear_3d(
            projection,
            inputs_embeds.cast(projection.weight.val().dtype()),
        )
    } else {
        inputs_embeds
    };
    let projected_inputs = projected_inputs.cast(
        predictor.model.layers[0]
            .self_attn
            .q_proj
            .weight
            .val()
            .dtype(),
    );

    let [batch_size, seq_len, _] = projected_inputs.dims();
    let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
    let device = projected_inputs.device();
    let attention_mask_shape = attention_mask.as_ref().map(Tensor::dims);
    let mask = build_attention_mask(batch_size, seq_len, key_len, attention_mask, &device);
    tracing::debug!(
        batch_size,
        seq_len,
        key_len,
        attention_mask_shape = ?attention_mask_shape,
        projected_dtype = ?projected_inputs.dtype(),
        "running code predictor layers"
    );

    predictor.model.run_layers(
        projected_inputs,
        config.code_predictor_config.num_attention_heads,
        config.code_predictor_config.num_key_value_heads,
        config.code_predictor_config.head_dim,
        &predictor.rope,
        mask,
        cache,
    )
}

fn batch_finished<B: Backend>(eos_mask: &Tensor<B, 1, burn::tensor::Bool>) -> bool {
    eos_mask
        .clone()
        .all()
        .into_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .expect("eos mask should be readable")[0]
}

fn last_hidden_step<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    hidden
        .slice([0..batch_size, seq_len - 1..seq_len, 0..hidden_size])
        .reshape([batch_size, hidden_size])
}

fn add_trailing_text_embed<B: Backend>(
    codec_embed: Tensor<B, 3>,
    trailing_text_hidden: Option<&Tensor<B, 3>>,
    tts_pad_embed: Option<&Tensor<B, 3>>,
    generation_step: usize,
) -> Tensor<B, 3> {
    match (trailing_text_hidden, tts_pad_embed) {
        (Some(trailing), Some(pad)) => {
            let [batch_size, trailing_len, hidden_size] = trailing.dims();
            if generation_step < trailing_len {
                codec_embed
                    + trailing.clone().slice([
                        0..batch_size,
                        generation_step..generation_step + 1,
                        0..hidden_size,
                    ])
            } else {
                codec_embed + pad.clone()
            }
        }
        _ => codec_embed,
    }
}

fn codec_group_context_embedding<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    codec_ids: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let [batch_size, _num_groups] = codec_ids.dims();
    let mut group_embeds = Vec::with_capacity(config.num_code_groups);
    let base_token = codec_ids
        .clone()
        .slice([0..batch_size, 0..1])
        .reshape([batch_size, 1]);
    group_embeds.push(
        loaded
            .model
            .talker
            .model
            .codec_embedding
            .forward(base_token),
    );
    for group_idx in 1..config.num_code_groups {
        let token = codec_ids
            .clone()
            .slice([0..batch_size, group_idx..group_idx + 1])
            .reshape([batch_size, 1]);
        group_embeds.push(
            loaded.model.talker.code_predictor.model.codec_embedding[group_idx - 1].forward(token),
        );
    }
    Tensor::cat(group_embeds, 1).sum_dim(1)
}

fn validate_generation_side_inputs(
    batch_size: usize,
    hidden_size: usize,
    trailing_dims: Option<[usize; 3]>,
    pad_dims: Option<[usize; 3]>,
) -> Result<(), Qwen3TtsInferenceError> {
    match (trailing_dims, pad_dims) {
        (None, None) => Ok(()),
        (
            Some([trailing_batch, trailing_len, trailing_hidden]),
            Some([pad_batch, pad_len, pad_hidden]),
        ) => {
            if trailing_batch != batch_size || pad_batch != batch_size {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input batch mismatch: expected {batch_size}, got trailing={trailing_batch}, pad={pad_batch}"
                    ),
                });
            }
            if trailing_len == 0 || pad_len != 1 {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input length mismatch: trailing length must be > 0 and pad length must be 1, got trailing={trailing_len}, pad={pad_len}"
                    ),
                });
            }
            if trailing_hidden != hidden_size || pad_hidden != hidden_size {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input hidden mismatch: expected {hidden_size}, got trailing={trailing_hidden}, pad={pad_hidden}"
                    ),
                });
            }
            Ok(())
        }
        _ => Err(Qwen3TtsInferenceError::InvalidInput {
            message: "trailing_text_hidden and tts_pad_embed must be provided together".to_string(),
        }),
    }
}

fn validate_code_predictor_cache_layer_count<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    cache: &[KeyValueCache<B>],
) -> Result<(), Qwen3TtsInferenceError> {
    let expected = config.code_predictor_config.num_hidden_layers;
    if cache.len() != expected {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "code predictor cache has {} layers but config expects {}",
                cache.len(),
                expected
            ),
        });
    }
    Ok(())
}

fn validate_cache_layer_count<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    cache: &[KeyValueCache<B>],
) -> Result<(), Qwen3TtsInferenceError> {
    if cache.len() != config.num_hidden_layers {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "cache has {} layers but talker config expects {}",
                cache.len(),
                config.num_hidden_layers
            ),
        });
    }
    Ok(())
}

fn validate_cache_lengths<B: Backend>(
    cache: &[KeyValueCache<B>],
) -> Result<usize, Qwen3TtsInferenceError> {
    let Some((first, rest)) = cache.split_first() else {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "decode cache must contain at least one layer".to_string(),
        });
    };
    let expected = first.len();
    for (idx, layer_cache) in rest.iter().enumerate() {
        let actual = layer_cache.len();
        if actual != expected {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "decode cache length mismatch at layer {}: expected {}, got {}",
                    idx + 1,
                    expected,
                    actual
                ),
            });
        }
    }
    Ok(expected)
}

fn validate_talker_input(
    name: &str,
    input_dims: [usize; 3],
    position_dims: [usize; 3],
    attention_dims: Option<[usize; 2]>,
    decode_cache_len: Option<usize>,
) -> Result<(), Qwen3TtsInferenceError> {
    let [batch_size, seq_len, _hidden_size] = input_dims;
    if batch_size == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("{name} batch size must be non-zero"),
        });
    }
    if seq_len == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("{name} sequence length must be non-zero"),
        });
    }
    if let Some(cache_len) = decode_cache_len {
        if seq_len != 1 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!("{name} expects exactly one token, got sequence length {seq_len}"),
            });
        }
        if cache_len == 0 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!("{name} requires a populated prefill cache"),
            });
        }
    }

    let expected_position_dims = [3, batch_size, seq_len];
    if position_dims != expected_position_dims {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "{name} position_ids shape mismatch: expected {:?}, got {:?}",
                expected_position_dims, position_dims
            ),
        });
    }

    if let Some([mask_batch_size, mask_seq_len]) = attention_dims {
        if mask_batch_size != batch_size {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "{name} attention_mask batch mismatch: expected {batch_size}, got {mask_batch_size}"
                ),
            });
        }

        let expected_mask_seq_len = decode_cache_len.map_or(seq_len, |cache_len| cache_len + 1);
        if mask_seq_len != expected_mask_seq_len {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "{name} attention_mask length mismatch: expected {expected_mask_seq_len}, got {mask_seq_len}"
                ),
            });
        }
    }

    Ok(())
}
