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
    let (last_hidden_state, logits) = loaded.model.talker.forward(
        input.inputs_embeds,
        input.position_ids,
        input.attention_mask,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        cache,
    );

    Ok(TalkerForwardOutput {
        last_hidden_state,
        logits,
        activations: BTreeMap::new(),
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
