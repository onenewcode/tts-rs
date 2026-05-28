use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::model::graph::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::model::graph::engine::components::generator::weights::LoadedQwen3TtsTalker;
use crate::runtime::kv::KeyValueCache;

use super::{
    TalkerStepOutput, validate_cache_layer_count, validate_cache_lengths, validate_talker_input,
};

pub(super) fn decode_step<B>(
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
