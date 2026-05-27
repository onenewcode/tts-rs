//! # Talker Inference Pipeline
//!
//! This module orchestrates the talker's autoregressive generation loop:
//!
//! ```text
//! Prefill (full sequence) → select first token → decode loop (one token at a time)
//!                                                      │
//!                          ┌───────────────────────────┘
//!                          │ 1. Embed selected token
//!                          │ 2. Run decode step (attention + KV cache)
//!                          │ 3. Apply sampling controls (V5)
//!                          │ 4. Apply repetition penalty (V6)
//!                          │ 5. Check EOS / max tokens
//!                          └→ next token or stop
//! ```
//!
//! ## Sampling Pipeline (sample_token)
//!
//! ```text
//! logits → suppress tokens → temperature → top-k → top-p → softmax → categorical
//! ```
//!
//! All sampling math uses Burn tensor operations (`gather`, `scatter`, `sort`,
//! `categorical`) and stays on-device.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::nn::mlp::native_linear_3d;

use crate::Qwen3TtsInferenceError;

use crate::shared::config::talker::Qwen3TtsTalkerConfig;
use crate::shared::io::talker_load::LoadedQwen3TtsTalker;
use crate::shared::runtime::cache::KeyValueCache;

// Re-exported from types.rs + shared/runtime/sampling.rs
pub use super::types::*;
use crate::shared::runtime::sampling::apply_repetition_penalty;
#[allow(unused_imports)]
pub use crate::shared::runtime::sampling::{SamplingConfig, StoppingRules, sample_token};

pub fn forward_talker_prefill<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerForwardInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerForwardOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    validate_talker_input(
        "talker prefill",
        input.inputs_embeds.dims(),
        input.position_ids.dims(),
        input.attention_mask.as_ref().map(Tensor::dims),
        None,
    )?;

    let (last_hidden_state, logits, activations, attention_activations) =
        loaded.model.talker.forward(
            input.inputs_embeds,
            input.position_ids,
            input.attention_mask,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            cache,
            input.collect_activations,
        );

    Ok(TalkerForwardOutput {
        last_hidden_state,
        logits,
        activations,
        attention_activations,
    })
}

pub fn forward_talker_decode_step<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerDecodeInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerDecodeOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    let cache_len = validate_cache_lengths(cache)?;
    validate_talker_input(
        "talker decode",
        input.inputs_embeds.dims(),
        input.position_ids.dims(),
        input.attention_mask.as_ref().map(Tensor::dims),
        Some(cache_len),
    )?;

    let (last_hidden_state, logits, activations, attention_activations) =
        loaded.model.talker.forward(
            input.inputs_embeds,
            input.position_ids,
            input.attention_mask,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            cache,
            input.collect_activations,
        );

    Ok(TalkerDecodeOutput {
        last_hidden_state,
        logits,
        activations,
        attention_activations,
    })
}

pub fn generate_talker_tokens<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerGenerateInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerGenerateOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    let max_new_tokens = input.stopping.max_new_tokens;
    validate_cache_layer_count(config, cache)?;
    if max_new_tokens == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "talker generation max_new_tokens must be greater than zero".to_string(),
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
        max_new_tokens,
        eos_token_id = ?input.stopping.eos_token_id,
        suppress_token_count = input.suppress_token_ids.len(),
        "starting talker token generation"
    );
    let collect_step_diagnostics = input.collect_step_diagnostics;
    let trailing_text_hidden = input.trailing_text_hidden;
    let tts_pad_embed = input.tts_pad_embed;
    validate_generation_side_inputs(
        batch_size,
        config.hidden_size,
        trailing_text_hidden.as_ref().map(Tensor::dims),
        tts_pad_embed.as_ref().map(Tensor::dims),
    )?;

    let prefill_output = forward_talker_prefill(
        config,
        loaded,
        TalkerForwardInput {
            inputs_embeds: input.prefill_inputs_embeds,
            position_ids: input.prefill_position_ids,
            attention_mask: input.prefill_attention_mask,
            collect_activations: false,
        },
        cache,
    )?;

    let sampling = &input.sampling;
    let eos_id = input.stopping.eos_token_id;
    let suppress = &input.suppress_token_ids;
    let rep_penalty = sampling.repetition_penalty;

    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits =
        apply_repetition_penalty(prefill_output.logits.clone(), &empty_history, rep_penalty);
    let (mut selected_token, mut eos_mask) =
        sample_token::<B>(prefill_logits, sampling, eos_id, suppress, &device);
    let mut generated_tokens: Vec<Tensor<B, 2, Int>> = vec![selected_token.clone()];
    let mut step_hidden_states = vec![last_hidden_step(prefill_output.last_hidden_state.clone())];
    let mut step_logits = Vec::new();
    let mut step_diagnostics = Vec::new();

    for _step_idx in 1..max_new_tokens {
        // If EOS set and all batch items stopped, exit early
        if eos_id.is_some()
            && eos_mask
                .clone()
                .all()
                .into_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap()[0]
        {
            break;
        }

        let cache_len_before = validate_cache_lengths(cache)?;
        let previous_hidden_state = step_hidden_states
            .last()
            .expect("generation always has a hidden state for the selected token")
            .clone();
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
        let codec_groups = generate_code_predictor_groups(
            config,
            loaded,
            CodePredictorGenerateInput {
                talker_hidden_state: previous_hidden_state,
                base_codec_token_id: selected_token,
                sampling: SamplingConfig::greedy(),
                collect_step_diagnostics: false,
            },
            &mut predictor_cache,
        )?;
        let inputs_embeds = codec_group_context_embedding(config, loaded, codec_groups.codec_ids);
        let inputs_embeds = add_trailing_text_embed(
            inputs_embeds,
            trailing_text_hidden.as_ref(),
            tts_pad_embed.as_ref(),
            generated_tokens.len() - 1,
        );
        let diagnostic_inputs_embeds = collect_step_diagnostics.then(|| inputs_embeds.clone());
        let position_ids =
            Tensor::<B, 3, Int>::full([3, batch_size, 1], cache_len_before as i32, &device);
        let decode_output = forward_talker_decode_step(
            config,
            loaded,
            TalkerDecodeInput {
                inputs_embeds,
                position_ids,
                attention_mask: None,
                collect_activations: collect_step_diagnostics,
            },
            cache,
        )?;

        let cache_len_after = validate_cache_lengths(cache)?;
        // Concatenate past tokens for repetition penalty
        let past_ids = Tensor::cat(generated_tokens.clone(), 1); // [batch, history]
        let penalized_logits =
            apply_repetition_penalty(decode_output.logits.clone(), &past_ids, rep_penalty);
        let (next_token, next_eos) =
            sample_token::<B>(penalized_logits, sampling, eos_id, suppress, &device);
        // EOS flag is sticky: once true, stays true
        eos_mask = eos_mask.bool_or(next_eos);
        selected_token = next_token;
        generated_tokens.push(selected_token.clone());
        step_hidden_states.push(last_hidden_step(decode_output.last_hidden_state.clone()));

        if collect_step_diagnostics {
            step_logits.push(decode_output.logits);
            let mut activations = decode_output.activations;
            if let Some(inputs_embeds) = diagnostic_inputs_embeds {
                activations.insert("decode.inputs_embeds".to_string(), inputs_embeds);
            }
            step_diagnostics.push(TalkerGenerateStepDiagnostic {
                cache_len_before,
                cache_len_after,
                activations,
                attention_activations: decode_output.attention_activations,
            });
        }
    }

    let generated_token_ids = Tensor::cat(generated_tokens, 1);
    tracing::info!(
        generated_tokens = generated_token_ids.dims()[1],
        hidden_steps = step_hidden_states.len(),
        "finished talker token generation"
    );

    Ok(TalkerGenerateOutput {
        generated_token_ids,
        step_hidden_states,
        prefill_logits: prefill_output.logits,
        step_logits,
        step_diagnostics,
    })
}

pub fn forward_code_predictor_teacher_forced<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: CodePredictorTeacherForcedInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<CodePredictorTeacherForcedOutput<B>, Qwen3TtsInferenceError>
where
    B: Backend,
{
    let predictor_config = &config.code_predictor_config;

    // Use pure operator-based logic for input embedding construction
    let [batch_size, _code_groups] = input.codec_ids.dims();
    let mut embeddings = Vec::with_capacity(config.num_code_groups);
    embeddings.push(input.talker_hidden_states.unsqueeze::<3>());

    for group_idx in 0..config.num_code_groups.saturating_sub(1) {
        let token_ids = input
            .codec_ids
            .clone()
            .slice([0..batch_size, group_idx..group_idx + 1])
            .reshape([batch_size, 1]);
        let embedding = if group_idx == 0 {
            loaded.model.talker.model.codec_embedding.forward(token_ids)
        } else {
            loaded.model.talker.code_predictor.model.codec_embedding[group_idx - 1]
                .forward(token_ids)
        };
        embeddings.push(embedding);
    }
    let inputs_embeds = Tensor::cat(embeddings, 1);

    let logits = loaded.model.talker.code_predictor.forward(
        inputs_embeds,
        predictor_config.num_attention_heads,
        predictor_config.num_key_value_heads,
        predictor_config.head_dim,
        input.attention_mask,
        cache,
    );

    Ok(CodePredictorTeacherForcedOutput {
        logits,
        activations: BTreeMap::new(),
        attention_activations: BTreeMap::new(),
    })
}

pub fn generate_code_predictor_groups<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: CodePredictorGenerateInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<CodePredictorGenerateOutput<B>, Qwen3TtsInferenceError>
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

    let [batch_size, hidden_size] = input.talker_hidden_state.dims();
    tracing::debug!(
        batch_size,
        hidden_size,
        code_groups = config.num_code_groups,
        "generating code predictor groups"
    );
    let base_token_dims = input.base_codec_token_id.dims();
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
        .forward(input.base_codec_token_id.clone());
    let prefill_inputs = Tensor::cat(
        vec![input.talker_hidden_state.unsqueeze::<3>(), base_embedding],
        1,
    );
    let (prefill_hidden, prefill_activations, prefill_attention_activations) =
        forward_code_predictor_hidden(
            config,
            loaded,
            prefill_inputs,
            None,
            cache,
            input.collect_step_diagnostics,
        );
    let prefill_head = &loaded.model.talker.code_predictor.lm_head[0];
    let prefill_logits = native_linear_3d(
        prefill_head,
        prefill_hidden
            .clone()
            .cast(prefill_head.weight.val().dtype()),
    );
    let device = prefill_logits.device();
    let sampling = &input.sampling;
    let eos_id: Option<usize> = None; // code predictor has no EOS — always generates N-1 tokens
    let suppress: &[usize] = &[];
    let rep_penalty = sampling.repetition_penalty;
    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits_pen =
        apply_repetition_penalty(prefill_logits.clone(), &empty_history, rep_penalty);
    let mut selected_token =
        sample_token::<B>(prefill_logits_pen, sampling, eos_id, suppress, &device).0;
    let mut predictor_tokens = vec![selected_token.clone()];
    let mut step_logits = Vec::new();
    let mut step_diagnostics = Vec::new();

    if input.collect_step_diagnostics {
        step_logits.push(prefill_logits);
        step_diagnostics.push(CodePredictorGenerateStepDiagnostic {
            cache_len_before: 0,
            cache_len_after: validate_cache_lengths(cache)?,
            activations: prefill_activations,
            attention_activations: prefill_attention_activations,
            cache_activations: cache_snapshots(cache),
        });
    }

    for head_idx in 1..config.num_code_groups - 1 {
        let cache_len_before = validate_cache_lengths(cache)?;
        let step_inputs = loaded.model.talker.code_predictor.model.codec_embedding[head_idx - 1]
            .forward(selected_token);
        let (step_hidden, step_activations, step_attention_activations) =
            forward_code_predictor_hidden(
                config,
                loaded,
                step_inputs,
                None,
                cache,
                input.collect_step_diagnostics,
            );
        let head = &loaded.model.talker.code_predictor.lm_head[head_idx];
        let logits = native_linear_3d(head, step_hidden.clone().cast(head.weight.val().dtype()));
        let cache_len_after = validate_cache_lengths(cache)?;
        let past_ids = Tensor::cat(predictor_tokens.clone(), 1); // [batch, history]
        let logits_pen = apply_repetition_penalty(logits.clone(), &past_ids, rep_penalty);
        let next_token = sample_token::<B>(logits_pen, sampling, eos_id, suppress, &device).0;
        selected_token = next_token;
        predictor_tokens.push(selected_token.clone());

        if input.collect_step_diagnostics {
            step_logits.push(logits);
            step_diagnostics.push(CodePredictorGenerateStepDiagnostic {
                cache_len_before,
                cache_len_after,
                activations: step_activations,
                attention_activations: step_attention_activations,
                cache_activations: cache_snapshots(cache),
            });
        }
    }

    let predictor_token_ids = Tensor::cat(predictor_tokens, 1);
    let codec_ids = Tensor::cat(
        vec![input.base_codec_token_id, predictor_token_ids.clone()],
        1,
    );
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
        cache_len = final_cache_len,
        codec_shape = ?codec_ids.dims(),
        "generated code predictor groups"
    );

    Ok(CodePredictorGenerateOutput {
        codec_ids,
        predictor_token_ids,
        step_logits,
        step_diagnostics,
    })
}

fn cache_snapshots<B: Backend>(cache: &[KeyValueCache<B>]) -> BTreeMap<String, Tensor<B, 4>> {
    let mut snapshots = BTreeMap::new();
    for (layer_idx, layer_cache) in cache.iter().enumerate() {
        if let Some(key) = layer_cache.key_snapshot() {
            snapshots.insert(format!("layers.{layer_idx}.cache.key"), key);
        }
        if let Some(value) = layer_cache.value_snapshot() {
            snapshots.insert(format!("layers.{layer_idx}.cache.value"), value);
        }
    }
    snapshots
}

fn forward_code_predictor_hidden<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    inputs_embeds: Tensor<B, 3>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    cache: &mut [KeyValueCache<B>],
    collect_activations: bool,
) -> (
    Tensor<B, 3>,
    BTreeMap<String, Tensor<B, 3>>,
    BTreeMap<String, Tensor<B, 4>>,
)
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
    let mask =
        super::model::build_attention_mask(batch_size, seq_len, key_len, attention_mask, &device);

    if collect_activations {
        predictor.model.forward_with_activations(
            projected_inputs,
            config.code_predictor_config.num_attention_heads,
            config.code_predictor_config.num_key_value_heads,
            config.code_predictor_config.head_dim,
            &predictor.rope,
            mask,
            cache,
        )
    } else {
        (
            predictor.model.forward(
                projected_inputs,
                config.code_predictor_config.num_attention_heads,
                config.code_predictor_config.num_key_value_heads,
                config.code_predictor_config.head_dim,
                &predictor.rope,
                mask,
                cache,
            ),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }
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
