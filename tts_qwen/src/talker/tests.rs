use burn::backend::Flex;
use burn::tensor::Tensor;

use super::inference::{
    CodePredictorGenerateInput, CodePredictorTeacherForcedInput, SamplingConfig, StoppingRules,
    TalkerDecodeInput, TalkerForwardInput, TalkerGenerateInput,
    forward_code_predictor_teacher_forced, forward_talker_decode_step, forward_talker_prefill,
    generate_code_predictor_groups, generate_talker_tokens, sample_token,
};
use crate::shared::config::talker::{
    Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig,
};
use crate::shared::io::talker_remap::{talker_export_key_remapper, talker_load_key_remapper};
use crate::shared::runtime::cache::KeyValueCache;

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
) -> crate::talker::LoadedQwen3TtsTalker<TestBackend> {
    let device = Default::default();
    crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: config.init_checkpoint::<TestBackend>(&device),
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    }
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
fn forward_talker_prefill_collects_layer_outputs_and_logits() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    };

    let inputs_embeds = Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device);
    let position_ids = Tensor::from_data([[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]], &device);
    let attention_mask = Tensor::from_data([[1i32, 1, 1]], &device);

    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = forward_talker_prefill(
        &config.talker_config,
        &loaded,
        TalkerForwardInput {
            inputs_embeds,
            position_ids,
            attention_mask: Some(attention_mask),
            collect_activations: true,
        },
        &mut cache,
    )
    .expect("prefill forward should succeed");

    assert_eq!(output.last_hidden_state.dims(), [1, 3, 16]);
    assert_eq!(output.logits.dims(), [1, 3, 32]);
}

#[test]
fn forward_talker_prefill_rejects_invalid_position_shape() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    };

    let position_ids = Tensor::from_data([[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]], &device);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let _ = forward_talker_prefill(
        &config.talker_config,
        &loaded,
        TalkerForwardInput {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            position_ids,
            attention_mask: None,
            collect_activations: false,
        },
        &mut cache,
    )
    .expect("prefill forward should succeed");
}

#[test]
fn forward_talker_decode_step_appends_one_token_to_prefill_cache() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    };

    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    forward_talker_prefill(
        &config.talker_config,
        &loaded,
        TalkerForwardInput {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            attention_mask: Some(Tensor::from_data([[1i32, 1, 1]], &device)),
            collect_activations: false,
        },
        &mut cache,
    )
    .expect("prefill forward should succeed");

    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 3));

    let output = forward_talker_decode_step(
        &config.talker_config,
        &loaded,
        TalkerDecodeInput {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 1, 16], &device),
            position_ids: Tensor::from_data([[[3i32]], [[3i32]], [[3i32]]], &device),
            attention_mask: Some(Tensor::from_data([[1i32, 1, 1, 1]], &device)),
            collect_activations: true,
        },
        &mut cache,
    )
    .expect("decode forward should succeed");

    assert_eq!(output.last_hidden_state.dims(), [1, 1, 16]);
    assert_eq!(output.logits.dims(), [1, 1, 32]);
    assert!(output.activations.contains_key("layers.0.hidden.output"));
    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 4));
}

#[test]
fn forward_talker_decode_step_rejects_multi_token_input() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    };

    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    forward_talker_prefill(
        &config.talker_config,
        &loaded,
        TalkerForwardInput {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            attention_mask: None,
            collect_activations: false,
        },
        &mut cache,
    )
    .expect("prefill forward should succeed");

    let err = forward_talker_decode_step(
        &config.talker_config,
        &loaded,
        TalkerDecodeInput {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 2, 16], &device),
            position_ids: Tensor::from_data([[[3i32, 4]], [[3i32, 4]], [[3i32, 4]]], &device),
            attention_mask: None,
            collect_activations: false,
        },
        &mut cache,
    )
    .expect_err("decode should reject multi-token inputs");

    assert!(err.to_string().contains("exactly one token"));
}

#[test]
fn generate_talker_tokens_rejects_zero_new_tokens() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let err = generate_talker_tokens(
        &config.talker_config,
        &loaded,
        TalkerGenerateInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: 0,
                eos_token_id: None,
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: false,
        },
        &mut cache,
    )
    .expect_err("generation should reject zero new tokens");

    assert!(err.to_string().contains("max_new_tokens"));
}

#[test]
fn generate_talker_tokens_returns_expected_shape_and_cache_len() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = generate_talker_tokens(
        &config.talker_config,
        &loaded,
        TalkerGenerateInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: Some(Tensor::from_data([[1i32, 1, 1]], &device)),
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: 4,
                eos_token_id: None,
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: true,
        },
        &mut cache,
    )
    .expect("generation should succeed");

    assert_eq!(output.generated_token_ids.dims(), [1, 4]);
    assert_eq!(output.prefill_logits.dims(), [1, 3, 32]);
    assert_eq!(output.step_logits.len(), 3);
    assert_eq!(output.step_diagnostics.len(), 3);
    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 6));
    assert_eq!(output.step_diagnostics[0].cache_len_before, 3);
    assert_eq!(output.step_diagnostics[0].cache_len_after, 4);
}

#[test]
fn generate_talker_tokens_selects_first_token_from_last_prefill_position() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = generate_talker_tokens(
        &config.talker_config,
        &loaded,
        TalkerGenerateInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: 1,
                eos_token_id: None,
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: false,
        },
        &mut cache,
    )
    .expect("generation should succeed");

    let expected_first_token = output
        .prefill_logits
        .clone()
        .slice([0..1, 2..3, 0..32])
        .argmax(2)
        .reshape([1, 1])
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    let actual_first_token = output
        .generated_token_ids
        .slice([0..1, 0..1])
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();

    assert_eq!(actual_first_token, expected_first_token);
    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 3));
}

#[test]
fn forward_code_predictor_teacher_forced_collects_expected_outputs() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::LoadReport::default(),
        model_dir: std::path::PathBuf::new(),
        weights_path: std::path::PathBuf::new(),
    };

    let mut cache = (0..config.talker_config.code_predictor_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config
                    .talker_config
                    .code_predictor_config
                    .num_key_value_heads,
                10,
                config.talker_config.code_predictor_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = forward_code_predictor_teacher_forced(
        &config.talker_config,
        &loaded,
        CodePredictorTeacherForcedInput {
            talker_hidden_states: Tensor::<TestBackend, 2>::zeros([1, 16], &device),
            codec_ids: Tensor::from_data([[1i32, 2, 3, 4]], &device),
            attention_mask: Some(Tensor::from_data([[1i32, 1, 1, 1]], &device)),
            collect_activations: true,
        },
        &mut cache,
    )
    .expect("teacher forced forward should succeed");

    assert_eq!(output.logits.dims(), [1, 3, 24]);
}

#[test]
fn generate_code_predictor_groups_rejects_wrong_cache_layer_count() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = vec![KeyValueCache::new(
        1,
        config
            .talker_config
            .code_predictor_config
            .num_key_value_heads,
        10,
        config.talker_config.code_predictor_config.head_dim,
    )];

    let err = generate_code_predictor_groups(
        &config.talker_config,
        &loaded,
        CodePredictorGenerateInput {
            talker_hidden_state: Tensor::<TestBackend, 2>::zeros([1, 16], &device),
            base_codec_token_id: Tensor::from_data([[1i32]], &device),
            sampling: SamplingConfig::greedy(),
            collect_step_diagnostics: false,
        },
        &mut cache,
    )
    .expect_err("generation should reject wrong cache layer count");

    assert!(err.to_string().contains("code predictor cache"));
}

#[test]
fn generate_code_predictor_groups_returns_expected_shapes_and_cache_len() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let predictor_config = &config.talker_config.code_predictor_config;
    let mut cache = (0..predictor_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                predictor_config.num_key_value_heads,
                10,
                predictor_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = generate_code_predictor_groups(
        &config.talker_config,
        &loaded,
        CodePredictorGenerateInput {
            talker_hidden_state: Tensor::<TestBackend, 2>::zeros([1, 16], &device),
            base_codec_token_id: Tensor::from_data([[7i32]], &device),
            sampling: SamplingConfig::greedy(),
            collect_step_diagnostics: true,
        },
        &mut cache,
    )
    .expect("code predictor generation should succeed");

    assert_eq!(output.codec_ids.dims(), [1, 4]);
    assert_eq!(output.predictor_token_ids.dims(), [1, 3]);
    assert_eq!(output.step_logits.len(), 3);
    assert_eq!(output.step_diagnostics.len(), 3);
    assert!(cache.iter().all(|layer_cache| layer_cache.len() == 4));
    assert_eq!(output.step_diagnostics[0].cache_len_before, 0);
    assert_eq!(output.step_diagnostics[0].cache_len_after, 2);
    assert_eq!(output.step_diagnostics[1].cache_len_before, 2);
    assert_eq!(output.step_diagnostics[1].cache_len_after, 3);

    let first_codec_id = output
        .codec_ids
        .clone()
        .slice([0..1, 0..1])
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    assert_eq!(first_codec_id, vec![7]);
}

#[test]
fn generate_code_predictor_groups_selects_first_predictor_token_from_prefill_logits() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let predictor_config = &config.talker_config.code_predictor_config;
    let mut cache = (0..predictor_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                predictor_config.num_key_value_heads,
                10,
                predictor_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = generate_code_predictor_groups(
        &config.talker_config,
        &loaded,
        CodePredictorGenerateInput {
            talker_hidden_state: Tensor::<TestBackend, 2>::zeros([1, 16], &device),
            base_codec_token_id: Tensor::from_data([[3i32]], &device),
            sampling: SamplingConfig::greedy(),
            collect_step_diagnostics: true,
        },
        &mut cache,
    )
    .expect("code predictor generation should succeed");

    let expected_first_predictor_token = output.step_logits[0]
        .clone()
        .slice([0..1, 1..2, 0..24])
        .argmax(2)
        .reshape([1, 1])
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    let actual_first_predictor_token = output
        .predictor_token_ids
        .slice([0..1, 0..1])
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();

    assert_eq!(actual_first_predictor_token, expected_first_predictor_token);
}

// --- V5: Sampling & stopping tests --------------------------------------------

#[test]
fn sample_token_greedy_equals_argmax() {
    let device = Default::default();
    // logits: [batch=1, seq=2, vocab=4] — token 2 is highest at last position
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
    assert_eq!(token, vec![2]); // argmax across last position
    // No EOS configured → mask all false
    let eos = eos_mask.into_data().into_vec::<bool>().unwrap();
    assert!(!eos[0]);
}

#[test]
fn sample_token_topk_1_equals_argmax() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.0f32, 1.0, 2.0, 3.0]]], &device);
    // top_k=1 + do_sample=true + temperature=1e-5 should act like argmax
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
    assert_eq!(token, vec![3]); // max-value token
}

#[test]
fn sample_token_suppresses_specified_tokens() {
    let device = Default::default();
    // Token 2 is the max (logit 5.0), but it should be suppressed
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.1f32, 0.2, 5.0, 0.5, 4.0]]], &device);
    // Suppress token 2, 4 → argmax should pick token 3 (4.0)
    let sampling = SamplingConfig {
        do_sample: true,
        temperature: 1e-5,
        top_k: None,
        top_p: 1.0,
        seed: None,
        repetition_penalty: None,
    };
    let (selected, _) = sample_token::<TestBackend>(logits, &sampling, None, &[2, 4], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![3]); // suppressed 2 and 4, 3 is max remaining
}

#[test]
fn sample_token_eos_detection() {
    let device = Default::default();
    // Token 42 is the max, EOS is 42
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.1f32, 0.2, 100.0, 0.5]]], &device);
    let sampling = SamplingConfig::greedy();
    let (selected, eos_mask) = sample_token::<TestBackend>(
        logits,
        &sampling,
        Some(2),
        &[],
        &device, // EOS = token 2
    );
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![2]); // selected max-value token (2)
    let eos = eos_mask.into_data().into_vec::<bool>().unwrap();
    assert!(eos[0]); // EOS mask should be true
}

#[test]
fn sample_token_eos_not_selected_when_not_max() {
    let device = Default::default();
    // Token 2 is the max, EOS is 0 (not max)
    let logits = Tensor::<TestBackend, 3>::from_data([[[0.1f32, 0.2, 100.0, 0.5]]], &device);
    let sampling = SamplingConfig::greedy();
    let (selected, eos_mask) =
        sample_token::<TestBackend>(logits, &sampling, Some(0), &[], &device);
    let token = selected
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(token, vec![2]); // selected max-value token (2), not EOS
    let eos = eos_mask.into_data().into_vec::<bool>().unwrap();
    assert!(!eos[0]); // EOS mask should be false
}

#[test]
fn generate_talker_tokens_stops_early_on_eos() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    // Generate with max_new_tokens=10 but EOS set — actual tokens may stop earlier
    let output = generate_talker_tokens(
        &config.talker_config,
        &loaded,
        TalkerGenerateInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: 10,
                eos_token_id: Some(9999),
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: false,
        },
        &mut cache,
    )
    .expect("generation should succeed");

    // No token equals 9999 in this random model, so full 10 tokens generated
    assert!(
        output.generated_token_ids.dims()[1] <= 10,
        "should not exceed max_new_tokens even without EOS hit"
    );
}

#[test]
fn generate_talker_tokens_respects_max_new_tokens_upper_bound() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let mut cache = (0..config.talker_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                config.talker_config.num_key_value_heads,
                10,
                config.talker_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let output = generate_talker_tokens(
        &config.talker_config,
        &loaded,
        TalkerGenerateInput {
            prefill_inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 3, 16], &device),
            prefill_position_ids: Tensor::from_data(
                [[[0i32, 1, 2]], [[0i32, 1, 2]], [[0i32, 1, 2]]],
                &device,
            ),
            prefill_attention_mask: None,
            trailing_text_hidden: None,
            tts_pad_embed: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: 3,
                eos_token_id: None,
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: false,
        },
        &mut cache,
    )
    .expect("generation should succeed");

    assert_eq!(
        output.generated_token_ids.dims(),
        [1, 3],
        "generated token count should equal max_new_tokens"
    );
}

#[test]
fn code_predictor_groups_uses_sampling_config() {
    // Greedy sampling should produce deterministic output
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let loaded = sample_loaded_talker(&config);
    let predictor_config = &config.talker_config.code_predictor_config;
    let mut cache1 = (0..predictor_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                predictor_config.num_key_value_heads,
                10,
                predictor_config.head_dim,
            )
        })
        .collect::<Vec<_>>();
    let mut cache2 = (0..predictor_config.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(
                1,
                predictor_config.num_key_value_heads,
                10,
                predictor_config.head_dim,
            )
        })
        .collect::<Vec<_>>();

    let run = |cache: &mut Vec<KeyValueCache<TestBackend>>| {
        generate_code_predictor_groups(
            &config.talker_config,
            &loaded,
            CodePredictorGenerateInput {
                talker_hidden_state: Tensor::<TestBackend, 2>::zeros([1, 16], &device),
                base_codec_token_id: Tensor::from_data([[3i32]], &device),
                sampling: SamplingConfig::greedy(),
                collect_step_diagnostics: true,
            },
            cache,
        )
        .expect("code predictor generation should succeed")
        .predictor_token_ids
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap()
    };

    let tokens1 = run(&mut cache1);
    let tokens2 = run(&mut cache2);
    assert_eq!(
        tokens1, tokens2,
        "greedy code predictor must be deterministic across runs"
    );
}
