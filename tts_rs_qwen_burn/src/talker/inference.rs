use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::Qwen3TtsInferenceError;

use super::cache::KeyValueCache;
use super::config::Qwen3TtsTalkerConfig;
use super::load::LoadedQwen3TtsTalker;

pub type TalkerActivations<B> = BTreeMap<String, Tensor<B, 3>>;

#[derive(Debug)]
pub struct TalkerForwardInput<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct TalkerForwardOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerDecodeInput<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct TalkerDecodeOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerGenerateInput<B: Backend> {
    pub prefill_inputs_embeds: Tensor<B, 3>,
    pub prefill_position_ids: Tensor<B, 3, Int>,
    pub prefill_attention_mask: Option<Tensor<B, 2, Int>>,
    pub max_new_tokens: usize,
    pub collect_step_diagnostics: bool,
}

#[derive(Debug)]
pub struct TalkerGenerateStepDiagnostic<B: Backend> {
    pub cache_len_before: usize,
    pub cache_len_after: usize,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerGenerateOutput<B: Backend> {
    pub generated_token_ids: Tensor<B, 2, Int>,
    pub prefill_logits: Tensor<B, 3>,
    pub step_logits: Vec<Tensor<B, 3>>,
    pub step_diagnostics: Vec<TalkerGenerateStepDiagnostic<B>>,
}

#[derive(Debug)]
pub struct CodePredictorTeacherForcedInput<B: Backend> {
    pub talker_hidden_states: Tensor<B, 2>,
    pub codec_ids: Tensor<B, 2, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct CodePredictorTeacherForcedOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

pub fn forward_talker_prefill<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerForwardInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerForwardOutput<B>, Qwen3TtsInferenceError> {
    validate_cache_layer_count(config, cache)?;
    validate_talker_input(
        "talker prefill",
        input.inputs_embeds.dims(),
        input.position_ids.dims(),
        input.attention_mask.as_ref().map(Tensor::dims),
        None,
    )?;

    let (last_hidden_state, logits, activations) = loaded.model.talker.forward(
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
    })
}

pub fn forward_talker_decode_step<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerDecodeInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerDecodeOutput<B>, Qwen3TtsInferenceError> {
    validate_cache_layer_count(config, cache)?;
    let cache_len = validate_cache_lengths(cache)?;
    validate_talker_input(
        "talker decode",
        input.inputs_embeds.dims(),
        input.position_ids.dims(),
        input.attention_mask.as_ref().map(Tensor::dims),
        Some(cache_len),
    )?;

    let (last_hidden_state, logits, activations) = loaded.model.talker.forward(
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
    })
}

pub fn generate_talker_tokens<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: TalkerGenerateInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerGenerateOutput<B>, Qwen3TtsInferenceError> {
    validate_cache_layer_count(config, cache)?;
    if input.max_new_tokens == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "talker generation max_new_tokens must be greater than zero".to_string(),
        });
    }

    for layer_cache in cache.iter_mut() {
        layer_cache.reset();
    }

    let device = input.prefill_inputs_embeds.device();
    let [batch_size, prefill_len, _hidden_size] = input.prefill_inputs_embeds.dims();
    let collect_step_diagnostics = input.collect_step_diagnostics;

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

    let mut selected_token = select_last_position_token(prefill_output.logits.clone());
    let mut generated_tokens = vec![selected_token.clone()];
    let mut step_logits = Vec::new();
    let mut step_diagnostics = Vec::new();

    for _step_idx in 1..input.max_new_tokens {
        let cache_len_before = validate_cache_lengths(cache)?;
        let inputs_embeds = loaded
            .model
            .talker
            .model
            .codec_embedding
            .forward(selected_token);
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
        selected_token = select_last_position_token(decode_output.logits.clone());
        generated_tokens.push(selected_token.clone());

        if collect_step_diagnostics {
            step_logits.push(decode_output.logits);
            step_diagnostics.push(TalkerGenerateStepDiagnostic {
                cache_len_before,
                cache_len_after,
                activations: decode_output.activations,
            });
        }
    }

    let generated_token_ids = Tensor::cat(generated_tokens, 1);
    let expected_cache_len = prefill_len + input.max_new_tokens - 1;
    let final_cache_len = validate_cache_lengths(cache)?;
    if final_cache_len != expected_cache_len {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "talker generation final cache length mismatch: expected {expected_cache_len}, got {final_cache_len}"
            ),
        });
    }

    Ok(TalkerGenerateOutput {
        generated_token_ids,
        prefill_logits: prefill_output.logits,
        step_logits,
        step_diagnostics,
    })
}

pub fn forward_code_predictor_teacher_forced<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    input: CodePredictorTeacherForcedInput<B>,
    cache: &mut [KeyValueCache<B>],
) -> Result<CodePredictorTeacherForcedOutput<B>, Qwen3TtsInferenceError> {
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
    })
}

fn select_last_position_token<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 2, Int> {
    let [batch_size, seq_len, _vocab_size] = logits.dims();
    logits
        .slice([0..batch_size, seq_len - 1..seq_len, 0.._vocab_size])
        .argmax(2)
        .reshape([batch_size, 1])
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
