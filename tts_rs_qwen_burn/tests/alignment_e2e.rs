mod common;

use std::process::Command;

use burn::backend::Flex;
use burn::tensor::{Int, Tensor};
use serde::Deserialize;
use tts_rs_qwen_burn::{
    CodePredictorGenerateInput, CustomVoiceBatch, CustomVoiceRequest, KeyValueCache,
    Qwen3TtsTextTokenizer, SamplingConfig, StoppingRules, TalkerGenerateInput,
    build_custom_voice_prefill_batch, decode_codec_tokens, generate_code_predictor_groups,
    generate_talker_tokens, load_custom_voice_generation_config, load_qwen3_tts_audio_codec,
    load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;

#[derive(Debug, Deserialize)]
struct TensorPreview {
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct E2eReference {
    text: String,
    language: String,
    speaker: String,
    base_token_ids: Vec<i32>,
    codec_groups: Vec<Vec<i32>>,
    codec_shape: Vec<usize>,
    talker_hidden: TensorPreview,
    waveform: TensorPreview,
}

fn top5(logits: Vec<f32>) -> Vec<(usize, f32)> {
    let mut indexed = logits.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(5);
    indexed
}

#[test]
#[ignore = "loads real talker/audio codec weights and runs the full Python oracle"]
fn e2e_matches_python_oracle() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_e2e.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_e2e.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--max-new-tokens",
            "7",
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python E2E oracle");
    assert!(status.success(), "Python E2E oracle failed");
    let reference: E2eReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();

    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device).unwrap();
    let audio_codec = load_qwen3_tts_audio_codec::<Backend>(&model_dir, &device).unwrap();
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();
    let generation_config = load_custom_voice_generation_config(&model_dir).unwrap();
    let frontend = build_custom_voice_prefill_batch(
        &tokenizer,
        &talker.config.talker_config,
        &talker,
        &CustomVoiceBatch::single(CustomVoiceRequest {
            text: reference.text,
            language: Some(reference.language),
            speaker: Some(reference.speaker),
        }),
        &device,
    )
    .unwrap();
    let cfg = &talker.config.talker_config;
    let mut talker_cache = (0..cfg.num_hidden_layers)
        .map(|_| KeyValueCache::new(1, cfg.num_key_value_heads, 4096, cfg.head_dim))
        .collect::<Vec<_>>();
    let generated = generate_talker_tokens(
        cfg,
        &talker,
        TalkerGenerateInput {
            prefill_inputs_embeds: frontend.inputs_embeds,
            prefill_position_ids: frontend.position_ids,
            prefill_attention_mask: Some(frontend.attention_mask),
            trailing_text_hidden: Some(frontend.trailing_text_hidden),
            tts_pad_embed: Some(frontend.tts_pad_embed),
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: reference.base_token_ids.len(),
                eos_token_id: Some(generation_config.codec_eos_token_id),
            },
            suppress_token_ids: generation_config.suppress_token_ids,
            collect_step_diagnostics: false,
        },
        &mut talker_cache,
    )
    .unwrap();
    let base_tokens = generated
        .generated_token_ids
        .clone()
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    assert_eq!(base_tokens, reference.base_token_ids);
    let hidden_steps = Tensor::cat(
        generated
            .step_hidden_states
            .iter()
            .map(|hidden| hidden.clone().unsqueeze::<3>())
            .collect(),
        1,
    );
    let hidden_values = hidden_steps
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let hidden_max_abs = hidden_values
        .iter()
        .zip(reference.talker_hidden.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let first_hidden_max_abs = hidden_values
        .iter()
        .take(cfg.hidden_size)
        .zip(reference.talker_hidden.values.iter().take(cfg.hidden_size))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let mut codec_steps = Vec::with_capacity(reference.base_token_ids.len());
    let mut rust_groups = Vec::with_capacity(reference.base_token_ids.len());
    let mut first_step_topk = Vec::new();
    for t in 0..reference.base_token_ids.len() {
        let base_token = generated
            .generated_token_ids
            .clone()
            .slice([0..1, t..t + 1]);
        let hidden = generated.step_hidden_states[t].clone();
        let mut predictor_cache = (0..cfg.code_predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    cfg.code_predictor_config.num_key_value_heads,
                    cfg.num_code_groups + 1,
                    cfg.code_predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();
        let groups = generate_code_predictor_groups(
            cfg,
            &talker,
            CodePredictorGenerateInput {
                talker_hidden_state: hidden,
                base_codec_token_id: base_token,
                sampling: SamplingConfig::greedy(),
                collect_step_diagnostics: t == 0,
            },
            &mut predictor_cache,
        )
        .unwrap();
        if t == 0 {
            first_step_topk = groups
                .step_logits
                .iter()
                .map(|logits| {
                    top5(
                        logits
                            .clone()
                            .into_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .unwrap(),
                    )
                })
                .collect();
        }
        rust_groups.push(
            groups
                .codec_ids
                .clone()
                .into_data()
                .convert::<i32>()
                .into_vec::<i32>()
                .unwrap(),
        );
        codec_steps.push(groups.codec_ids.reshape([1, cfg.num_code_groups, 1]));
    }
    assert_eq!(
        rust_groups, reference.codec_groups,
        "talker hidden preview max_abs={hidden_max_abs}, first_step_max_abs={first_hidden_max_abs}; first Rust predictor topk: {first_step_topk:?}"
    );
    let codec_tokens: Tensor<Backend, 3, Int> = Tensor::cat(codec_steps, 2);
    assert_eq!(
        codec_tokens.dims().as_slice(),
        reference.codec_shape.as_slice()
    );
    let waveform = decode_codec_tokens::<Backend>(
        &audio_codec,
        codec_tokens,
        &audio_codec.config.decoder_config,
    )
    .unwrap();
    let actual_wave = waveform
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(actual_wave.len() >= reference.waveform.values.len());
    let max_abs = actual_wave
        .iter()
        .zip(reference.waveform.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= 1e-1, "waveform preview max_abs={max_abs}");
}
