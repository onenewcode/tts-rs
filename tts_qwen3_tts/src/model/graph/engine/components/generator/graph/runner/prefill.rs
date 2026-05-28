use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::model::graph::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::model::graph::engine::components::generator::weights::LoadedQwen3TtsTalker;
use crate::runtime::kv::KeyValueCache;

use super::{TalkerStepOutput, validate_cache_layer_count, validate_talker_input};

pub(super) fn prefill<B>(
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
