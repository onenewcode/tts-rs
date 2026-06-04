use burn::nn::attention::generate_autoregressive_mask;
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use self::sampling::{
    SamplingConfig, apply_repetition_penalty,
    repetition_penalty_enabled as is_repetition_penalty_enabled, sample_token,
    sample_token_with_suppress_mask, suppress_token_mask,
};
use super::network::kv::KeyValueCache;
use crate::error::QwenTtsInferenceError;
use crate::model::talker::config::Qwen3TtsTalkerConfig;
use crate::model::talker::weights::LoadedQwen3TtsTalker;

pub mod sampling;

#[derive(Debug)]
pub struct TalkerGenerator<B: Backend> {
    config: Qwen3TtsTalkerConfig,
    decode_cache: Vec<KeyValueCache<B>>,
    predictor_cache: Vec<KeyValueCache<B>>,
    current_hidden: Option<Tensor<B, 2>>,
    selected_token: Option<Tensor<B, 2, Int>>,
    eos_seen: bool,
    past_token_ids: Option<Tensor<B, 2, Int>>,
    codec_steps: Vec<Tensor<B, 3, Int>>,
    trailing_text_hidden: Tensor<B, 3>,
    trailing_text_len: usize,
    tts_pad_embed: Tensor<B, 3>,
    sampling: SamplingConfig,
    suppress_mask: Option<Tensor<B, 2, Bool>>,
    repetition_penalty_enabled: bool,
    max_new_tokens: usize,
    eos_token_id: Option<i64>,
    step_idx: usize,
    finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TalkerStep {
    pub finished: bool,
}

#[derive(Debug)]
pub struct TalkerGenerationOutput<B: Backend> {
    pub codec_token_ids: Tensor<B, 3, Int>,
}

pub struct TalkerGeneratorStart<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub trailing_text_hidden: Tensor<B, 3>,
    pub tts_pad_embed: Tensor<B, 3>,
    pub sampling: SamplingConfig,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<i64>,
    pub suppress_token_ids: Vec<usize>,
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
        init: TalkerGeneratorStart<B>,
    ) -> Result<Self, QwenTtsInferenceError> {
        let TalkerGeneratorStart {
            inputs_embeds,
            position_ids,
            attention_mask,
            trailing_text_hidden,
            tts_pad_embed,
            sampling,
            max_new_tokens,
            eos_token_id,
            suppress_token_ids,
        } = init;
        if max_new_tokens == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "talker max_new_tokens must be greater than zero".to_string(),
            });
        }
        let [batch_size, _, _] = inputs_embeds.dims();
        if batch_size != 1 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!("only batch size 1 is supported, got {batch_size}"),
            });
        }
        let [trailing_batch, trailing_len, trailing_hidden] = trailing_text_hidden.dims();
        let [pad_batch, pad_len, pad_hidden] = tts_pad_embed.dims();
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
        if trailing_hidden != config.hidden_size || pad_hidden != config.hidden_size {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "generation side input hidden mismatch: expected {}, got trailing={}, pad={}",
                    config.hidden_size, trailing_hidden, pad_hidden
                ),
            });
        }

        let mut decode_cache = (0..config.num_hidden_layers)
            .map(|_| KeyValueCache::new(1, config.num_key_value_heads, 4096, config.head_dim))
            .collect::<Vec<_>>();
        for layer_cache in &mut decode_cache {
            layer_cache.reset();
        }
        let predictor_cache = (0..config.code_predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    config.code_predictor_config.num_key_value_heads,
                    config.num_code_groups + 1,
                    config.code_predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();

        let prefill = prefill(
            config,
            loaded,
            inputs_embeds,
            position_ids,
            Some(attention_mask),
            &mut decode_cache,
        )?;
        let current_hidden = last_hidden_step(prefill.last_hidden_state);
        // Repetition penalty only applies once we have generated history.
        let penalized_logits = prefill.logits;
        let [logits_batch, _, logits_vocab] = penalized_logits.dims();
        let suppress_mask = suppress_token_mask::<B>(
            logits_batch,
            logits_vocab,
            &suppress_token_ids,
            &penalized_logits.device(),
        );
        let selected_token = sample_token_with_suppress_mask::<B>(
            penalized_logits,
            &sampling,
            suppress_mask.as_ref(),
        );
        let eos_seen = selected_token_is_eos(&selected_token, eos_token_id)?;
        let repetition_penalty_enabled = is_repetition_penalty_enabled(sampling.repetition_penalty);
        let past_token_ids = repetition_penalty_enabled.then(|| selected_token.clone());

        Ok(Self {
            config: config.clone(),
            decode_cache,
            predictor_cache,
            current_hidden: Some(current_hidden),
            selected_token: Some(selected_token),
            eos_seen,
            past_token_ids,
            codec_steps: Vec::new(),
            trailing_text_hidden,
            trailing_text_len: trailing_len,
            tts_pad_embed,
            sampling,
            suppress_mask,
            repetition_penalty_enabled,
            max_new_tokens,
            eos_token_id,
            step_idx: 0,
            finished: false,
        })
    }

    pub fn step(
        &mut self,
        loaded: &LoadedQwen3TtsTalker<B>,
    ) -> Result<Option<TalkerStep>, QwenTtsInferenceError> {
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

        let current_hidden = self
            .current_hidden
            .take()
            .expect("talker current hidden state should be present while stepping");
        let selected_token = self
            .selected_token
            .take()
            .expect("talker selected token should be present while stepping");
        let past_token_ids = self.past_token_ids.take();
        let codec_ids = generate_code_predictor_groups(
            &self.config,
            loaded,
            current_hidden,
            selected_token,
            &self.sampling,
            &mut self.predictor_cache,
        )?;
        let generation_step = self.step_idx;
        let inputs_embeds = add_trailing_text_embed(
            codec_group_context_embedding(&self.config, loaded, codec_ids.clone()),
            &self.trailing_text_hidden,
            self.trailing_text_len,
            generation_step,
            &self.tts_pad_embed,
        );
        self.codec_steps
            .push(codec_ids.reshape([1, self.config.num_code_groups, 1]));
        self.step_idx += 1;

        if self.step_idx >= self.max_new_tokens {
            self.finished = true;
            return Ok(Some(TalkerStep { finished: true }));
        }

        let cache_len = self
            .decode_cache
            .first()
            .ok_or_else(|| QwenTtsInferenceError::InvalidInput {
                message: "decode cache must contain at least one layer".to_string(),
            })?
            .len();
        let device = inputs_embeds.device();
        let cache_len =
            i64::try_from(cache_len).map_err(|_| QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "decode cache length {cache_len} does not fit the model int tensor"
                ),
            })?;
        let position_ids = Tensor::<B, 3, Int>::full([3, 1, 1], cache_len, &device);
        let decoded = decode_step(
            &self.config,
            loaded,
            inputs_embeds,
            position_ids,
            &mut self.decode_cache,
        )?;
        let penalized_logits = if self.repetition_penalty_enabled {
            apply_repetition_penalty(
                decoded.logits,
                past_token_ids.as_ref().expect(
                    "talker token history should be present when repetition penalty is enabled",
                ),
                self.sampling.repetition_penalty,
            )
        } else {
            decoded.logits
        };
        let next_token = sample_token_with_suppress_mask::<B>(
            penalized_logits,
            &self.sampling,
            self.suppress_mask.as_ref(),
        );
        self.selected_token = Some(next_token.clone());
        self.eos_seen = self.eos_seen || selected_token_is_eos(&next_token, self.eos_token_id)?;
        self.past_token_ids = if self.repetition_penalty_enabled {
            Some(Tensor::cat(
                vec![
                    past_token_ids.expect(
                        "talker token history should be present when repetition penalty is enabled",
                    ),
                    next_token,
                ],
                1,
            ))
        } else {
            None
        };
        self.current_hidden = Some(last_hidden_step(decoded.last_hidden_state));
        if self.eos_seen {
            self.finished = true;
        }
        Ok(Some(TalkerStep {
            finished: self.finished,
        }))
    }

    pub fn finalize(self) -> Result<TalkerGenerationOutput<B>, QwenTtsInferenceError> {
        if self.codec_steps.is_empty() {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "no codec tokens were generated".to_string(),
            });
        }
        Ok(TalkerGenerationOutput {
            codec_token_ids: Tensor::cat(self.codec_steps, 2),
        })
    }
}

fn selected_token_is_eos<B: Backend>(
    selected_token: &Tensor<B, 2, Int>,
    eos_token_id: Option<i64>,
) -> Result<bool, QwenTtsInferenceError> {
    let Some(id) = eos_token_id else {
        return Ok(false);
    };
    let token_id = selected_token
        .clone()
        .try_into_scalar()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("talker.selected_token_is_eos: {source}"),
        })?
        .elem::<i64>();
    Ok(token_id == id)
}

pub(crate) fn last_hidden_step<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, _seq_len, hidden_size] = hidden.dims();
    select_last_sequence_step(hidden).reshape([batch_size, hidden_size])
}

pub(super) fn select_last_sequence_step<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    if seq_len == 1 {
        return hidden;
    }
    hidden.slice([0..batch_size, seq_len - 1..seq_len, 0..hidden_size])
}

pub(crate) fn add_trailing_text_embed<B: Backend>(
    codec_embed: Tensor<B, 3>,
    trailing_text_hidden: &Tensor<B, 3>,
    trailing_text_len: usize,
    generation_step: usize,
    tts_pad_embed: &Tensor<B, 3>,
) -> Tensor<B, 3> {
    if generation_step < trailing_text_len {
        let [batch_size, _seq_len, hidden_size] = trailing_text_hidden.dims();
        codec_embed
            + trailing_text_hidden.clone().slice([
                0..batch_size,
                generation_step..generation_step + 1,
                0..hidden_size,
            ])
    } else {
        codec_embed + tts_pad_embed.clone()
    }
}

pub(crate) fn codec_group_context_embedding<B: Backend>(
    config: &Qwen3TtsTalkerConfig,
    loaded: &LoadedQwen3TtsTalker<B>,
    codec_ids: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let mut codec_groups = codec_ids.chunk(config.num_code_groups, 1).into_iter();
    let base_token = codec_groups
        .next()
        .expect("codec ids should include the base group");
    let mut summed = loaded
        .model
        .talker
        .model
        .codec_embedding
        .forward(base_token);
    for (group_idx, token) in codec_groups.enumerate() {
        summed = summed
            + loaded.model.talker.code_predictor.model.codec_embedding[group_idx].forward(token);
    }
    summed
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
    debug_assert_eq!(cache.len(), config.num_hidden_layers);

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
    debug_assert_eq!(cache.len(), config.num_hidden_layers);

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
    if config.num_code_groups < 2 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: "code predictor generation requires at least two code groups".to_string(),
        });
    }
    debug_assert_eq!(cache.len(), config.code_predictor_config.num_hidden_layers);
    for layer_cache in cache.iter_mut() {
        layer_cache.reset_preserve_allocation();
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
    let prefill_logits = prefill_head.forward(prefill_hidden);
    let mut selected_token = Some(sample_token::<B>(prefill_logits, sampling, &[]));
    if config.num_code_groups == 2 {
        return Ok(Tensor::cat(
            vec![
                base_codec_token_id,
                selected_token
                    .take()
                    .expect("prefill should produce the first code predictor token"),
            ],
            1,
        ));
    }
    let mut predictor_token_ids = selected_token
        .as_ref()
        .expect("prefill should produce the first code predictor token")
        .clone();

    for head_idx in 1..config.num_code_groups - 1 {
        let step_inputs = loaded.model.talker.code_predictor.model.codec_embedding[head_idx - 1]
            .forward(
                selected_token
                    .take()
                    .expect("predictor token should be present for the next head"),
            );
        let step_hidden = run_code_predictor_hidden(config, loaded, step_inputs, None, cache);
        let head = &loaded.model.talker.code_predictor.lm_head[head_idx];
        let logits = apply_repetition_penalty(
            head.forward(step_hidden),
            &predictor_token_ids,
            sampling.repetition_penalty,
        );
        let next_token = sample_token::<B>(logits, sampling, &[]);
        predictor_token_ids = if head_idx + 1 == config.num_code_groups - 1 {
            Tensor::cat(vec![predictor_token_ids, next_token], 1)
        } else {
            selected_token = Some(next_token.clone());
            Tensor::cat(vec![predictor_token_ids, next_token], 1)
        };
    }

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
    let causal_mask = (seq_len == key_len).then(|| {
        generate_autoregressive_mask::<B>(batch_size, seq_len, &device).unsqueeze_dim::<4>(1)
    });
    let padding_mask =
        attention_mask.map(|mask| mask.equal_elem(0).unsqueeze::<4>().repeat_dim(2, seq_len));
    let mask = match (causal_mask, padding_mask) {
        (Some(causal), Some(padding)) => Some(causal.bool_or(padding)),
        (Some(causal), None) => Some(causal),
        (None, Some(padding)) => Some(padding),
        (None, None) => None,
    };

    predictor.model.forward(
        projected_inputs,
        config.code_predictor_config.num_attention_heads,
        config.code_predictor_config.num_key_value_heads,
        config.code_predictor_config.head_dim,
        &predictor.rope,
        mask.as_ref(),
        cache,
    )
}
