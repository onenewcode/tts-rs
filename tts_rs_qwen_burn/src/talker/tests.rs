use burn::backend::Flex;

use super::config::{Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};
use super::remap::{talker_export_key_remapper, talker_load_key_remapper};

type TestBackend = Flex;

fn sample_talker_config(
    code_predictor_hidden_size: usize,
    num_code_groups: usize,
) -> Qwen3TtsConfig {
    Qwen3TtsConfig::new(Qwen3TtsTalkerConfig::new(
        Qwen3TtsTalkerCodePredictorConfig::new(
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
        ),
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
