use burn::backend::Flex;
use burn::tensor::Tensor;

use super::cache::KeyValueCache;
use super::config::{Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};
use super::inference::{
    CodePredictorTeacherForcedInput, TalkerForwardInput, forward_code_predictor_teacher_forced,
    forward_talker_prefill,
};
use super::remap::{talker_export_key_remapper, talker_load_key_remapper};

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
        load_report: crate::manifest::LoadReport::default(),
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
                &device,
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
        load_report: crate::manifest::LoadReport::default(),
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
                &device,
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
fn forward_code_predictor_teacher_forced_collects_expected_outputs() {
    let config = sample_talker_config(12, 4);
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let loaded = crate::talker::LoadedQwen3TtsTalker {
        config: config.clone(),
        model: checkpoint,
        load_report: crate::manifest::LoadReport::default(),
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
                &device,
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
