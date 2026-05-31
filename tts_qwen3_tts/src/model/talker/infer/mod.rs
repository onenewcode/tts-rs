use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use self::sampling::{SamplingConfig, apply_repetition_penalty, sample_token};
use super::network::kv::KeyValueCache;
use crate::error::QwenTtsInferenceError;
use crate::execution::compiler::session_seed::SessionSeed;
use crate::model::nn::sequence::select_last_sequence_step;
use crate::model::talker::config::Qwen3TtsTalkerConfig;
use crate::model::talker::network::build_attention_mask;
use crate::model::talker::weights::LoadedQwen3TtsTalker;

pub mod sampling;

#[derive(Debug)]
pub struct TalkerGenerator<B: Backend> {
    config: Qwen3TtsTalkerConfig,
    decode_cache: Vec<KeyValueCache<B>>,
    current_hidden: Tensor<B, 2>,
    selected_token: Tensor<B, 2, Int>,
    eos_seen: bool,
    talker_tokens: Vec<Tensor<B, 2, Int>>,
    codec_steps: Vec<Tensor<B, 3, Int>>,
    trailing_text_hidden: Option<Tensor<B, 3>>,
    tts_pad_embed: Option<Tensor<B, 3>>,
    sampling: SamplingConfig,
    max_new_tokens: usize,
    eos_token_id: Option<usize>,
    suppress_token_ids: Vec<usize>,
    step_idx: usize,
    finished: bool,
}

#[derive(Debug)]
pub struct TalkerStep<B: Backend> {
    pub finished: bool,
    _codec_ids: Tensor<B, 2, Int>,
}

#[derive(Debug)]
pub struct TalkerGenerationOutput<B: Backend> {
    pub codec_token_ids: Tensor<B, 3, Int>,
}

pub(crate) struct TalkerStepOutput<B: Backend> {
    pub(crate) last_hidden_state: Tensor<B, 3>,
    pub(crate) logits: Tensor<B, 3>,
}

impl<B> TalkerGenerator<B>
where
    B: Backend,
{
    pub fn start(
        config: &Qwen3TtsTalkerConfig,
        loaded: &LoadedQwen3TtsTalker<B>,
        compiled: &SessionSeed<B>,
        sampling: SamplingConfig,
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        suppress_token_ids: Vec<usize>,
    ) -> Result<Self, QwenTtsInferenceError> {
        if max_new_tokens == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "talker max_new_tokens must be greater than zero".to_string(),
            });
        }
        let [batch_size, _, _] = compiled.inputs_embeds.dims();
        if batch_size != 1 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!("only batch size 1 is supported, got {batch_size}"),
            });
        }
        validate_generation_side_inputs(
            batch_size,
            config.hidden_size,
            Some(compiled.trailing_text_hidden.dims()),
            Some(compiled.tts_pad_embed.dims()),
        )?;

        let mut decode_cache = (0..config.num_hidden_layers)
            .map(|_| KeyValueCache::new(1, config.num_key_value_heads, 4096, config.head_dim))
            .collect::<Vec<_>>();
        for layer_cache in &mut decode_cache {
            layer_cache.reset();
        }

        let prefill = prefill(
            config,
            loaded,
            compiled.inputs_embeds.clone(),
            compiled.position_ids.clone(),
            Some(compiled.attention_mask.clone()),
            &mut decode_cache,
        )?;
        let current_hidden = last_hidden_step(prefill.last_hidden_state.clone());
        // Repetition penalty only applies once we have generated history.
        let penalized_logits = prefill.logits.clone();
        let selected_token = sample_token::<B>(penalized_logits, &sampling, &suppress_token_ids);
        let eos_seen = selected_token_is_eos(&selected_token, eos_token_id)?;

        Ok(Self {
            config: config.clone(),
            decode_cache,
            current_hidden,
            selected_token: selected_token.clone(),
            eos_seen,
            talker_tokens: vec![selected_token],
            codec_steps: Vec::new(),
            trailing_text_hidden: Some(compiled.trailing_text_hidden.clone()),
            tts_pad_embed: Some(compiled.tts_pad_embed.clone()),
            sampling,
            max_new_tokens,
            eos_token_id,
            suppress_token_ids,
            step_idx: 0,
            finished: false,
        })
    }

    pub fn step(
        &mut self,
        loaded: &LoadedQwen3TtsTalker<B>,
    ) -> Result<Option<TalkerStep<B>>, QwenTtsInferenceError> {
        if self.finished {
            return Ok(None);
        }
        if self.eos_seen {
            if self.codec_steps.is_empty() {
                return Err(QwenTtsInferenceError::InvalidInput {
                    message: "talker emitted EOS before any audio codec token".to_string(),
                });
            }
            self.finished = true;
            return Ok(None);
        }

        let mut predictor_cache = (0..self.config.code_predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    self.config.code_predictor_config.num_key_value_heads,
                    self.config.num_code_groups + 1,
                    self.config.code_predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();
        let codec_ids = generate_code_predictor_groups(
            &self.config,
            loaded,
            self.current_hidden.clone(),
            self.selected_token.clone(),
            &self.sampling,
            &mut predictor_cache,
        )?;
        self.codec_steps.push(
            codec_ids
                .clone()
                .reshape([1, self.config.num_code_groups, 1]),
        );
        self.step_idx += 1;

        if self.step_idx >= self.max_new_tokens {
            self.finished = true;
            return Ok(Some(TalkerStep {
                finished: true,
                _codec_ids: codec_ids,
            }));
        }

        let cache_len = validate_cache_lengths(&self.decode_cache)?;
        let inputs_embeds = add_trailing_text_embed(
            codec_group_context_embedding(&self.config, loaded, codec_ids.clone()),
            self.trailing_text_hidden.as_ref(),
            self.tts_pad_embed.as_ref(),
            self.step_idx - 1,
        );
        let device = inputs_embeds.device();
        let position_ids = Tensor::<B, 3, Int>::full([3, 1, 1], cache_len as i32, &device);
        let past_ids = Tensor::cat(self.talker_tokens.clone(), 1);
        let decoded = decode_step(
            &self.config,
            loaded,
            inputs_embeds,
            position_ids,
            &mut self.decode_cache,
        )?;
        let penalized_logits =
            apply_repetition_penalty(decoded.logits, &past_ids, self.sampling.repetition_penalty);
        let next_token =
            sample_token::<B>(penalized_logits, &self.sampling, &self.suppress_token_ids);
        self.selected_token = next_token.clone();
        self.eos_seen = self.eos_seen || selected_token_is_eos(&next_token, self.eos_token_id)?;
        self.talker_tokens.push(next_token);
        self.current_hidden = last_hidden_step(decoded.last_hidden_state);
        if self.eos_seen {
            self.finished = true;
        }
        Ok(Some(TalkerStep {
            finished: self.finished,
            _codec_ids: codec_ids,
        }))
    }

    pub fn finalize(&self) -> Result<TalkerGenerationOutput<B>, QwenTtsInferenceError> {
        if self.codec_steps.is_empty() {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "no codec tokens were generated".to_string(),
            });
        }
        Ok(TalkerGenerationOutput {
            codec_token_ids: Tensor::cat(self.codec_steps.clone(), 2),
        })
    }
}

fn selected_token_is_eos<B: Backend>(
    selected_token: &Tensor<B, 2, Int>,
    eos_token_id: Option<usize>,
) -> Result<bool, QwenTtsInferenceError> {
    let Some(id) = eos_token_id else {
        return Ok(false);
    };
    let is_eos = selected_token
        .clone()
        .equal_elem(id as i64)
        .float()
        .reshape([1])
        .try_into_scalar()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("talker.selected_token_is_eos: {source}"),
        })?
        .elem::<f32>();
    Ok(is_eos > 0.5)
}

pub(crate) fn last_hidden_step<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, _seq_len, hidden_size] = hidden.dims();
    select_last_sequence_step(hidden).reshape([batch_size, hidden_size])
}

pub(crate) fn add_trailing_text_embed<B: Backend>(
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

pub(crate) fn codec_group_context_embedding<B: Backend>(
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

pub(crate) fn validate_generation_side_inputs(
    batch_size: usize,
    hidden_size: usize,
    trailing_dims: Option<[usize; 3]>,
    pad_dims: Option<[usize; 3]>,
) -> Result<(), QwenTtsInferenceError> {
    match (trailing_dims, pad_dims) {
        (None, None) => Ok(()),
        (
            Some([trailing_batch, trailing_len, trailing_hidden]),
            Some([pad_batch, pad_len, pad_hidden]),
        ) => {
            if trailing_batch != batch_size || pad_batch != batch_size {
                return Err(QwenTtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input batch mismatch: expected {batch_size}, got trailing={trailing_batch}, pad={pad_batch}"
                    ),
                });
            }
            if trailing_len == 0 || pad_len != 1 {
                return Err(QwenTtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input length mismatch: trailing length must be > 0 and pad length must be 1, got trailing={trailing_len}, pad={pad_len}"
                    ),
                });
            }
            if trailing_hidden != hidden_size || pad_hidden != hidden_size {
                return Err(QwenTtsInferenceError::InvalidInput {
                    message: format!(
                        "generation side input hidden mismatch: expected {hidden_size}, got trailing={trailing_hidden}, pad={pad_hidden}"
                    ),
                });
            }
            Ok(())
        }
        _ => Err(QwenTtsInferenceError::InvalidInput {
            message: "trailing_text_hidden and tts_pad_embed must be provided together".to_string(),
        }),
    }
}

pub(crate) fn validate_code_predictor_cache_layer_count<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    cache: &[KeyValueCache<B>],
) -> Result<(), QwenTtsInferenceError> {
    let expected = config.code_predictor_config.num_hidden_layers;
    if cache.len() != expected {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "code predictor cache has {} layers but config expects {}",
                cache.len(),
                expected
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_cache_layer_count<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    cache: &[KeyValueCache<B>],
) -> Result<(), QwenTtsInferenceError> {
    if cache.len() != config.num_hidden_layers {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "cache has {} layers but talker config expects {}",
                cache.len(),
                config.num_hidden_layers
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_cache_lengths<B: Backend>(
    cache: &[KeyValueCache<B>],
) -> Result<usize, QwenTtsInferenceError> {
    let Some((first, rest)) = cache.split_first() else {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: "decode cache must contain at least one layer".to_string(),
        });
    };
    let expected = first.len();
    for (idx, layer_cache) in rest.iter().enumerate() {
        let actual = layer_cache.len();
        if actual != expected {
            return Err(QwenTtsInferenceError::InvalidInput {
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
pub(crate) fn validate_talker_input(
    name: &str,
    input_dims: [usize; 3],
    position_dims: [usize; 3],
    attention_dims: Option<[usize; 2]>,
    decode_cache_len: Option<usize>,
) -> Result<(), QwenTtsInferenceError> {
    let [batch_size, seq_len, _hidden_size] = input_dims;
    if batch_size == 0 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!("{name} batch size must be non-zero"),
        });
    }
    if seq_len == 0 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!("{name} sequence length must be non-zero"),
        });
    }
    if let Some(cache_len) = decode_cache_len {
        if seq_len != 1 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!("{name} expects exactly one token, got sequence length {seq_len}"),
            });
        }
        if cache_len == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!("{name} requires a populated prefill cache"),
            });
        }
    }

    let expected_position_dims = [3, batch_size, seq_len];
    if position_dims != expected_position_dims {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "{name} position_ids shape mismatch: expected {:?}, got {:?}",
                expected_position_dims, position_dims
            ),
        });
    }

    if let Some([mask_batch_size, mask_seq_len]) = attention_dims {
        if mask_batch_size != batch_size {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "{name} attention_mask batch mismatch: expected {batch_size}, got {mask_batch_size}"
                ),
            });
        }

        let expected_mask_seq_len = decode_cache_len.map_or(seq_len, |cache_len| cache_len + 1);
        if mask_seq_len != expected_mask_seq_len {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "{name} attention_mask length mismatch: expected {expected_mask_seq_len}, got {mask_seq_len}"
                ),
            });
        }
    }

    Ok(())
}

fn prefill<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    inputs_embeds: Tensor<B, 3>,
    position_ids: Tensor<B, 3, Int>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    cache: &mut [KeyValueCache<B>],
) -> Result<TalkerStepOutput<B>, QwenTtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    validate_talker_input(
        "talker prefill",
        inputs_embeds.dims(),
        position_ids.dims(),
        attention_mask.as_ref().map(Tensor::dims),
        None,
    )?;

    let (last_hidden_state, logits) = loaded.model.talker.forward(
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
) -> Result<TalkerStepOutput<B>, QwenTtsInferenceError>
where
    B: Backend,
{
    validate_cache_layer_count(config, cache)?;
    let cache_len = validate_cache_lengths(cache)?;
    validate_talker_input(
        "talker decode",
        inputs_embeds.dims(),
        position_ids.dims(),
        None,
        Some(cache_len),
    )?;

    let (last_hidden_state, logits) = loaded.model.talker.forward(
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

fn generate_code_predictor_groups<B>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    talker_hidden_state: Tensor<B, 2>,
    base_codec_token_id: Tensor<B, 2, Int>,
    sampling: &SamplingConfig,
    cache: &mut [KeyValueCache<B>],
) -> Result<Tensor<B, 2, Int>, QwenTtsInferenceError>
where
    B: Backend,
{
    validate_code_predictor_cache_layer_count(config, cache)?;
    if config.num_code_groups < 2 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: "code predictor generation requires at least two code groups".to_string(),
        });
    }

    for layer_cache in cache.iter_mut() {
        layer_cache.reset();
    }

    let [batch_size, hidden_size] = talker_hidden_state.dims();
    let base_token_dims = base_codec_token_id.dims();
    if base_token_dims != [batch_size, 1] {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "code predictor base token shape mismatch: expected {:?}, got {:?}",
                [batch_size, 1],
                base_token_dims
            ),
        });
    }
    if hidden_size != config.hidden_size {
        return Err(QwenTtsInferenceError::InvalidInput {
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
    let prefill_logits = prefill_head.forward(
        prefill_hidden
            .clone()
            .cast(prefill_head.weight.val().dtype()),
    );
    let mut selected_token = sample_token::<B>(prefill_logits, sampling, &[]);
    let mut predictor_tokens = vec![selected_token.clone()];

    for head_idx in 1..config.num_code_groups - 1 {
        let step_inputs = loaded.model.talker.code_predictor.model.codec_embedding[head_idx - 1]
            .forward(selected_token);
        let step_hidden = run_code_predictor_hidden(config, loaded, step_inputs, None, cache);
        let head = &loaded.model.talker.code_predictor.lm_head[head_idx];
        let past_ids = Tensor::cat(predictor_tokens.clone(), 1);
        let logits = apply_repetition_penalty(
            head.forward(step_hidden.clone().cast(head.weight.val().dtype())),
            &past_ids,
            sampling.repetition_penalty,
        );
        selected_token = sample_token::<B>(logits, sampling, &[]);
        predictor_tokens.push(selected_token.clone());
    }

    let predictor_token_ids = Tensor::cat(predictor_tokens, 1);
    Ok(Tensor::cat(
        vec![base_codec_token_id, predictor_token_ids],
        1,
    ))
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
        projection.forward(inputs_embeds)
    } else {
        inputs_embeds
    };

    let [batch_size, seq_len, _] = projected_inputs.dims();
    let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
    let device = projected_inputs.device();
    let mask = build_attention_mask(batch_size, seq_len, key_len, attention_mask, &device);

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
