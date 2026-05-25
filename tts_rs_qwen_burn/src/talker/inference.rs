use std::collections::BTreeMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData};

use super::nn::mlp::native_linear_3d;

use crate::Qwen3TtsInferenceError;

use super::cache::KeyValueCache;
use super::config::Qwen3TtsTalkerConfig;
use super::load::LoadedQwen3TtsTalker;

pub type TalkerActivations<B> = BTreeMap<String, Tensor<B, 3>>;

// --- V5: Sampling and Stopping ---

/// Controls how tokens are selected from logits during generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// `false` = greedy argmax (deterministic). `true` = apply temperature / top_k / top_p.
    pub do_sample: bool,
    /// Softmax temperature, clamped to `[1e-5, inf)`. Default 1.0.
    pub temperature: f32,
    /// Keep only the top-k logits before softmax. `None` = no truncation.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep minimum token set with cumulative prob ≥ top_p. 1.0 = off.
    pub top_p: f32,
    /// PRNG seed for reproducibility. `None` = non-deterministic.
    pub seed: Option<u64>,
    /// Repetition penalty. Values < 1.0 discourage repeated tokens, > 1.0 encourage.
    /// `None` = off (no penalty). Applied before temperature/top-k/top-p.
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: None,
            repetition_penalty: None,
        }
    }
}

impl SamplingConfig {
    /// Convenience: greedy mode with all sampling knobs off.
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            ..Default::default()
        }
    }
}

/// Conditions that cause autoregressive generation to stop.
#[derive(Debug, Clone)]
pub struct StoppingRules {
    /// Hard cap on how many tokens to generate (prefill length excluded).
    pub max_new_tokens: usize,
    /// Stop early when this token is selected. `None` = no early termination.
    pub eos_token_id: Option<usize>,
}

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
    pub sampling: SamplingConfig,
    pub stopping: StoppingRules,
    pub suppress_token_ids: Vec<usize>,
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

#[derive(Debug)]
pub struct CodePredictorGenerateInput<B: Backend> {
    pub talker_hidden_state: Tensor<B, 2>,
    pub base_codec_token_id: Tensor<B, 2, Int>,
    pub sampling: SamplingConfig,
    pub collect_step_diagnostics: bool,
}

#[derive(Debug)]
pub struct CodePredictorGenerateStepDiagnostic {
    pub cache_len_before: usize,
    pub cache_len_after: usize,
}

#[derive(Debug)]
pub struct CodePredictorGenerateOutput<B: Backend> {
    pub codec_ids: Tensor<B, 2, Int>,
    pub predictor_token_ids: Tensor<B, 2, Int>,
    pub step_logits: Vec<Tensor<B, 3>>,
    pub step_diagnostics: Vec<CodePredictorGenerateStepDiagnostic>,
}

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
    let [batch_size, _prefill_len, _hidden_size] = input.prefill_inputs_embeds.dims();
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

    let sampling = &input.sampling;
    let eos_id = input.stopping.eos_token_id;
    let suppress = &input.suppress_token_ids;
    let rep_penalty = sampling.repetition_penalty;

    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits = apply_repetition_penalty_3d(
        prefill_output.logits.clone(), &empty_history, rep_penalty,
    );
    let (mut selected_token, mut eos_mask) =
        sample_token::<B>(prefill_logits, sampling, eos_id, suppress, &device);
    let mut generated_tokens: Vec<Tensor<B, 2, Int>> = vec![selected_token.clone()];
    let mut step_logits = Vec::new();
    let mut step_diagnostics = Vec::new();

    for _step_idx in 1..max_new_tokens {
        // If EOS set and all batch items stopped, exit early
        if eos_id.is_some() && eos_mask.clone().all().into_data().convert::<bool>().into_vec::<bool>().unwrap()[0] {
            break;
        }

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
        // Concatenate past tokens for repetition penalty
        let past_ids = Tensor::cat(generated_tokens.clone(), 1); // [batch, history]
        let penalized_logits = apply_repetition_penalty_3d(
            decode_output.logits.clone(), &past_ids, rep_penalty,
        );
        let (next_token, next_eos) =
            sample_token::<B>(penalized_logits, sampling, eos_id, suppress, &device);
        // EOS flag is sticky: once true, stays true
        eos_mask = eos_mask.bool_or(next_eos);
        selected_token = next_token;
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

    Ok(TalkerGenerateOutput {
        generated_token_ids,
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
    let prefill_hidden = forward_code_predictor_hidden(
        config,
        loaded,
        prefill_inputs,
        None,
        cache,
    );
    let prefill_logits = native_linear_3d(&loaded.model.talker.code_predictor.lm_head[0], prefill_hidden);
    let device = prefill_logits.device();
    let sampling = &input.sampling;
    let eos_id: Option<usize> = None; // code predictor has no EOS — always generates N-1 tokens
    let suppress: &[usize] = &[];
    let rep_penalty = sampling.repetition_penalty;
    let empty_history = Tensor::<B, 2, Int>::zeros([batch_size, 0], &device);
    let prefill_logits_pen =
        apply_repetition_penalty_3d(prefill_logits.clone(), &empty_history, rep_penalty);
    let (mut selected_token, _) =
        sample_token::<B>(prefill_logits_pen, sampling, eos_id, suppress, &device);
    let mut predictor_tokens = vec![selected_token.clone()];
    let mut step_logits = Vec::new();
    let mut step_diagnostics = Vec::new();

    if input.collect_step_diagnostics {
        step_logits.push(prefill_logits);
        step_diagnostics.push(CodePredictorGenerateStepDiagnostic {
            cache_len_before: 0,
            cache_len_after: validate_cache_lengths(cache)?,
        });
    }

    for head_idx in 1..config.num_code_groups - 1 {
        let cache_len_before = validate_cache_lengths(cache)?;
        let step_inputs = loaded.model.talker.code_predictor.model.codec_embedding[head_idx - 1]
            .forward(selected_token);
        let step_hidden =
            forward_code_predictor_hidden(config, loaded, step_inputs, None, cache);
        let logits = native_linear_3d(&loaded.model.talker.code_predictor.lm_head[head_idx], step_hidden);
        let cache_len_after = validate_cache_lengths(cache)?;
        let past_ids = Tensor::cat(predictor_tokens.clone(), 1); // [batch, history]
        let logits_pen = apply_repetition_penalty_3d(logits.clone(), &past_ids, rep_penalty);
        let (next_token, _) = sample_token::<B>(logits_pen, sampling, eos_id, suppress, &device);
        selected_token = next_token;
        predictor_tokens.push(selected_token.clone());

        if input.collect_step_diagnostics {
            step_logits.push(logits);
            step_diagnostics.push(CodePredictorGenerateStepDiagnostic {
                cache_len_before,
                cache_len_after,
            });
        }
    }

    let predictor_token_ids = Tensor::cat(predictor_tokens, 1);
    let codec_ids = Tensor::cat(vec![input.base_codec_token_id, predictor_token_ids.clone()], 1);
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
        predictor_config.num_code_groups,
        config.num_code_groups,
        "talker and code predictor code group counts should match"
    );

    Ok(CodePredictorGenerateOutput {
        codec_ids,
        predictor_token_ids,
        step_logits,
        step_diagnostics,
    })
}

fn forward_code_predictor_hidden<B>(
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
        projection.forward(inputs_embeds)
    } else {
        inputs_embeds
    };

    let [batch_size, seq_len, _] = projected_inputs.dims();
    let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
    let device = projected_inputs.device();
    let mask = super::model::build_attention_mask(
        batch_size,
        seq_len,
        key_len,
        attention_mask,
        &device,
    );

    predictor.model.forward(
        projected_inputs,
        config.code_predictor_config.num_attention_heads,
        config.code_predictor_config.num_key_value_heads,
        config.code_predictor_config.head_dim,
        &predictor.rope,
        mask,
        cache,
    )
}

// -- V5: Sampling ---------------------------------------------------------------

/// Select one token per batch item from the last position of logits.
///
/// Greedy mode (`do_sample = false`): equivalent to argmax, bit-identical to the
/// old `select_last_position_token` helper.
///
/// Sampling mode (`do_sample = true`): applies
/// suppress → temperature → top-k → top-p → softmax → categorical.
///
/// Returns `(selected_token_ids, eos_mask)`:
/// - `selected_token_ids`: `[batch, 1]`
/// - `eos_mask`: `[batch]`, true where the selected token equals `eos_token_id`
pub fn sample_token<B: Backend>(
    logits: Tensor<B, 3>,
    sampling: &SamplingConfig,
    eos_token_id: Option<usize>,
    suppress_token_ids: &[usize],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Bool>) {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let last_logits = logits
        .slice([0..batch_size, seq_len - 1..seq_len, 0..vocab_size])
        .reshape([batch_size, vocab_size]); // [batch, vocab]

    if !sampling.do_sample {
        let selected = last_logits.argmax(1).reshape([batch_size, 1]); // [batch, 1]
        let eos_mask = match eos_token_id {
            Some(id) => selected
                .clone()
                .equal_elem(id as i64)
                .reshape([batch_size]),
            None => Tensor::<B, 1, Bool>::zeros([batch_size], device),
        };
        return (selected, eos_mask);
    }

    // --- Sampling path ---

    let mut logits_2d = last_logits;

    // 1. Suppress tokens — build mask on host, upload once
    if !suppress_token_ids.is_empty() {
        let mut mask_data = vec![false; batch_size * vocab_size];
        for batch in 0..batch_size {
            for &id in suppress_token_ids {
                if id < vocab_size {
                    mask_data[batch * vocab_size + id] = true;
                }
            }
        }
        let suppress_mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, [batch_size, vocab_size]),
            device,
        );
        logits_2d = logits_2d.mask_fill(suppress_mask, f32::NEG_INFINITY);
    }

    // 2. Temperature
    let temperature = sampling.temperature.max(1e-5);
    logits_2d = logits_2d.div_scalar(temperature);

    // 3. Top-k
    if let Some(k) = sampling.top_k {
        if k > 0 && k < vocab_size {
            let kth_value = logits_2d
                .clone()
                .topk(k, 1)
                .slice([0..batch_size, k - 1..k]);
            let mask = logits_2d.clone().lower(kth_value);
            logits_2d = logits_2d.mask_fill(mask, f32::NEG_INFINITY);
        }
    }

    // 4. Top-p (nucleus sampling)
    if sampling.top_p < 1.0 {
        let (sorted_vals, sorted_idx) = logits_2d.clone().sort_descending_with_indices(1);
        let sorted_probs = softmax(sorted_vals.clone().cast(DType::F32), 1).cast(DType::F32);
        let cumsum = sorted_probs.clone().cumsum(1);
        // Keep token i iff the cumulative probability BEFORE token i is < top_p,
        // i.e. (cumsum - probs) < top_p. The first token that crosses top_p is kept.
        let sorted_keep: Tensor<B, 2, Bool> = cumsum
            .sub(sorted_probs)
            .lower_elem(sampling.top_p);
        // Unsort the boolean mask back to original vocab order via gather
        let inverse = sorted_idx.argsort(1);
        let orig_keep: Tensor<B, 2, Bool> = sorted_keep.gather(1, inverse);
        logits_2d = logits_2d.mask_fill(orig_keep.bool_not(), f32::NEG_INFINITY);
    }

    // 5. Softmax + categorical sample
    let probs = softmax(logits_2d.clone().cast(DType::F32), 1);
    let selected = probs.categorical(1); // [batch, 1]

    let eos_mask = match eos_token_id {
        Some(id) => selected
            .clone()
            .equal_elem(id as i64)
            .reshape([batch_size]),
        None => Tensor::<B, 1, Bool>::zeros([batch_size], device),
    };

    (selected, eos_mask)
}

// -- Repetition penalty -------------------------------------------------------

/// Apply repetition penalty to logits: `logits[:, token_id] /= penalty` for each
/// past token. Uses gather + scatter-add for a pure Burn implementation.
fn apply_repetition_penalty_3d<B: Backend>(
    logits: Tensor<B, 3>,
    past_token_ids: &Tensor<B, 2, Int>,
    penalty: Option<f32>,
) -> Tensor<B, 3> {
    let Some(penalty) = penalty else { return logits };
    if penalty == 1.0 {
        return logits;
    }
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let history_len = past_token_ids.dims()[1];
    if history_len == 0 {
        return logits;
    }
    // Work in 2D: [batch, vocab]
    let logits_2d = logits.reshape([batch_size, vocab_size]);
    // Gather logits at past token positions: [batch, history]
    let gathered = logits_2d.clone().gather(1, past_token_ids.clone());
    // delta = orig * (1/penalty - 1); scatter-add delta → logit/penalty
    let scale = 1.0 / penalty - 1.0;
    let deltas = gathered.mul_scalar(scale);
    let result_2d = logits_2d.scatter(1, past_token_ids.clone(), deltas, IndexingUpdateOp::Add);
    result_2d.reshape([batch_size, seq_len, vocab_size])
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
