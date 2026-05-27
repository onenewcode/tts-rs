use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::kernels::mlp::native_linear_3d;
use crate::model::config::talker::Qwen3TtsTalkerConfig;
use crate::model::load::talker::LoadedQwen3TtsTalker;
use crate::model::qwen_tts::build_attention_mask;
use crate::runtime::kv::KeyValueCache;
use crate::runtime::sampling::{SamplingConfig, apply_repetition_penalty, sample_token};

use super::validate_code_predictor_cache_layer_count;

pub(super) fn infer_code_predictor_groups<B>(
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
    let prefill_inputs = Tensor::cat(vec![talker_hidden_state.unsqueeze::<3>(), base_embedding], 1);
    let prefill_hidden = run_code_predictor_hidden(config, loaded, prefill_inputs, None, cache);
    let prefill_head = &loaded.model.talker.code_predictor.lm_head[0];
    let prefill_logits = native_linear_3d(
        prefill_head,
        prefill_hidden.clone().cast(prefill_head.weight.val().dtype()),
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
    }

    let predictor_token_ids = Tensor::cat(predictor_tokens, 1);
    Ok(Tensor::cat(vec![base_codec_token_id, predictor_token_ids], 1))
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
        native_linear_3d(projection, inputs_embeds.cast(projection.weight.val().dtype()))
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
    let mask = build_attention_mask(batch_size, seq_len, key_len, attention_mask, &device);

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
