use burn::backend::Flex;
use burn::tensor::Tensor;

use super::inference::{TalkerInferInput, infer};
use crate::shared::config::talker::{
    Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig,
};
use crate::shared::io::talker_remap::{talker_export_key_remapper, talker_load_key_remapper};
use crate::shared::runtime::cache::KeyValueCache;
use crate::shared::runtime::sampling::{SamplingConfig, sample_token};

type TestBackend = Flex;

fn sample_talker_config(
    code_predictor_hidden_size: usize,
    num_code_groups: usize,
) -> Qwen3TtsConfig {
    let predictor_config = Qwen3TtsTalkerCodePredictorConfig::new(
        24,
        code_predictor_hidden_size,
        48,
        2,
        3,
        1,
        4,
        1e-5,
        false,
        num_code_groups,
    );

    Qwen3TtsConfig::new(Qwen3TtsTalkerConfig::new(
        predictor_config,
        32,
        16,
        32,
        2,
        4,
        2,
        4,
        1e-5,
        true,
        num_code_groups,
        8,
        20,
    ))
}

fn apply_remapper(remapper: &burn_store::KeyRemapper, key: &str) -> String {
    let mut out = key.to_string();
    for (pattern, replacement) in &remapper.patterns {
        if pattern.is_match(&out) {
            out = pattern.replace_all(&out, replacement.as_str()).to_string();
        }
    }
    out
}

fn sample_loaded_talker(
    config: &Qwen3TtsConfig,
) -> crate::shared::io::talker_load::LoadedQwen3TtsTalker<TestBackend> {
    let device = Default::default();
    crate::shared::io::talker_load::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: config.init_checkpoint::<TestBackend>(&device),
        load_report: crate::LoadReport::default(),
    }
}

fn sample_cache(config: &Qwen3TtsTalkerConfig) -> Vec<KeyValueCache<TestBackend>> {
    (0..config.num_hidden_layers)
        .map(|_| KeyValueCache::new(1, config.num_key_value_heads, 10, config.head_dim))
        .collect::<Vec<_>>()
}

#[test]
fn init_checkpoint_uses_configured_layer_counts_and_projection_dims() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let talker = checkpoint.talker;

    assert_eq!(talker.model.layers.len(), 2);
    assert_eq!(talker.code_predictor.model.layers.len(), 2);
    assert_eq!(talker.code_predictor.lm_head.len(), 3);
    assert!(talker.code_predictor.small_to_mtp_projection.is_some());
    assert_eq!(talker.model.codec_embedding.weight.dims(), [32, 16]);
    assert_eq!(
        talker.model.layers[0].self_attn.q_proj.weight.dims(),
        [16, 16]
    );
    assert_eq!(
        talker.model.layers[0].self_attn.k_proj.weight.dims(),
        [16, 8]
    );
    assert_eq!(
        talker.code_predictor.model.codec_embedding[0].weight.dims(),
        [24, 16]
    );
    assert_eq!(talker.text_projection.linear_fc2.weight.dims(), [8, 16]);
}

#[test]
fn init_checkpoint_omits_projection_when_hidden_sizes_match() {
    let config = sample_talker_config(16, 2);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);

    assert!(
        checkpoint
            .talker
            .code_predictor
            .small_to_mtp_projection
            .is_none()
    );
    assert_eq!(checkpoint.talker.code_predictor.lm_head.len(), 1);
}

#[test]
fn code_predictor_heads_saturate_at_zero() {
    let config = sample_talker_config(16, 0);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);

    assert!(checkpoint.talker.code_predictor.lm_head.is_empty());
    assert!(
        checkpoint
            .talker
            .code_predictor
            .model
            .codec_embedding
            .is_empty()
    );
}

#[test]
fn talker_load_remapper_maps_norm_weight_to_gamma() {
    let remapped = apply_remapper(&talker_load_key_remapper(), "talker.model.norm.weight");
    assert_eq!(remapped, "talker.model.norm.gamma");
}

#[test]
fn talker_export_remapper_maps_norm_gamma_to_weight() {
    let remapped = apply_remapper(&talker_export_key_remapper(), "talker.model.norm.gamma");
    assert_eq!(remapped, "talker.model.norm.weight");
}

#[test]
fn infer_rejects_zero_new_tokens() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = sample_cache(&config.talker_config);

    let err = infer(
        &config.talker_config,
        &loaded,
        TalkerInferInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 0,
            eos_token_id: None,
            suppress_token_ids: vec![],
        },
        &mut cache,
    )
    .expect_err("infer should reject zero new tokens");

    assert!(err.to_string().contains("max_new_tokens"));
}

#[test]
fn infer_returns_expected_token_shapes_and_cache_len() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = sample_cache(&config.talker_config);

    let output = infer(
        &config.talker_config,
        &loaded,
        TalkerInferInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: Some(Tensor::from_data([[1i32, 1, 1]], &device)),
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 4,
            eos_token_id: None,
            suppress_token_ids: vec![],
        },
        &mut cache,
    )
    .expect("infer should succeed");

    assert_eq!(output.talker_token_ids.dims(), [1, 4]);
    assert_eq!(output.codec_token_ids.dims(), [1, 4, 4]);
    assert_eq!(output.generated_audio_steps, 4);
    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 6));
}

#[test]
fn infer_requires_paired_generation_side_inputs() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = sample_cache(&config.talker_config);

    let err = infer(
        &config.talker_config,
        &loaded,
        TalkerInferInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: Some(Tensor::<TestBackend, 3>::zeros([1, 1, 16], &device)),
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 1,
            eos_token_id: None,
            suppress_token_ids: vec![],
        },
        &mut cache,
    )
    .expect_err("infer should require paired side inputs");

    assert!(err.to_string().contains("must be provided together"));
}

#[test]
fn infer_respects_max_new_tokens_upper_bound() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = sample_cache(&config.talker_config);

    let output = infer(
        &config.talker_config,
        &loaded,
        TalkerInferInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 3,
            eos_token_id: None,
            suppress_token_ids: vec![],
        },
        &mut cache,
    )
    .expect("infer should succeed");

    assert_eq!(output.talker_token_ids.dims(), [1, 3]);
    assert_eq!(output.generated_audio_steps, 3);
}

#[test]
fn sample_token_greedy_equals_argmax() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_data(
        [[[0.1f32, 0.2, 0.3, 0.4], [1.0, 2.0, 5.0, 0.5]]],
        &device,
    );
    let sampling = SamplingConfig::greedy();
    let (selected, eos_mask) = sample_token::<TestBackend>(logits, &sampling, None, &[], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![2]);
    let eos = eos_mask.into_data().into_vec::<bool>().unwrap();
    assert!(!eos[0]);
}

#[test]
fn sample_token_topk_1_equals_argmax() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.0f32, 1.0, 2.0, 3.0]]], &device);
    let sampling = SamplingConfig {
        do_sample: true,
        temperature: 1e-5,
        top_k: Some(1),
        top_p: 1.0,
        seed: None,
        repetition_penalty: None,
    };
    let (selected, _) = sample_token::<TestBackend>(logits, &sampling, None, &[], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![3]);
}

#[test]
fn sample_token_suppresses_specified_tokens() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.1f32, 0.2, 5.0, 4.0, 0.5]]], &device);
    let sampling = SamplingConfig {
        do_sample: true,
        temperature: 1e-5,
        top_k: None,
        top_p: 1.0,
        seed: None,
        repetition_penalty: None,
    };
    let (selected, _) = sample_token::<TestBackend>(logits, &sampling, None, &[2], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![3]);
}

#[test]
fn sample_token_eos_detection() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.1f32, 0.2, 100.0, 0.5]]], &device);
    let sampling = SamplingConfig::greedy();
    let (selected, eos_mask) =
        sample_token::<TestBackend>(logits, &sampling, Some(2), &[], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![2]);
    let eos = eos_mask.into_data().into_vec::<bool>().unwrap();
    assert!(eos[0]);
}
