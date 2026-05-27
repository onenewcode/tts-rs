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
    waveform_stats: Option<ReferenceWaveStats>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct WaveStats {
    min: f32,
    max: f32,
    peak: f32,
    rms: f32,
    clip_fraction: f32,
}

#[derive(Debug, Deserialize)]
struct ReferenceWaveStats {
    peak: f32,
    rms: f32,
    clip_fraction: f32,
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
    let mut all_step_topk = Vec::new();
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
                collect_step_diagnostics: true,
            },
            &mut predictor_cache,
        )
        .unwrap();
        let topk = groups
            .step_logits
            .iter()
            .map(|logits| top5_last_position(logits.clone()))
            .collect::<Vec<_>>();
        if t == 0 {
            first_step_topk = topk.clone();
        }
        all_step_topk.push(topk);
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
    let codec_mismatch = codec_group_mismatch_summary(&rust_groups, &reference.codec_groups);
    let first_mismatch_topk =
        first_mismatch_topk_report(&rust_groups, &reference.codec_groups, &all_step_topk);
    assert_eq!(
        rust_groups, reference.codec_groups,
        "talker hidden preview max_abs={hidden_max_abs}, first_step_max_abs={first_hidden_max_abs}; {codec_mismatch}; waveform preview max_abs={max_abs}; first Rust predictor topk: {first_step_topk:?}; first mismatch Rust topk: {first_mismatch_topk}"
    );
    assert!(max_abs <= 1e-1, "waveform preview max_abs={max_abs}");
}

#[test]
#[ignore = "loads real audio codec weights and decodes Python eager codec groups"]
fn rust_audio_codec_decodes_python_eager_codes_without_clipping() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_e2e_audio_codec.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_e2e.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--text",
            "你好，欢迎使用语音合成。",
            "--language",
            "Chinese",
            "--speaker",
            "Vivian",
            "--max-new-tokens",
            "64",
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python E2E oracle");
    assert!(status.success(), "Python E2E oracle failed");
    let reference: E2eReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();

    let device = Default::default();
    let audio_codec = load_qwen3_tts_audio_codec::<Backend>(&model_dir, &device).unwrap();
    let time_steps = reference.codec_groups.len();
    let num_groups = reference
        .codec_groups
        .first()
        .expect("reference should have codec groups")
        .len();
    let mut flat = Vec::with_capacity(time_steps * num_groups);
    for group_idx in 0..num_groups {
        for step in &reference.codec_groups {
            flat.push(step[group_idx]);
        }
    }
    let codec_tokens = Tensor::<Backend, 3, Int>::from_data(
        burn::tensor::TensorData::new(flat, [1, num_groups, time_steps]),
        &device,
    );
    let waveform = decode_codec_tokens::<Backend>(
        &audio_codec,
        codec_tokens,
        &audio_codec.config.decoder_config,
    )
    .unwrap();
    let actual_preview = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let preview_max_abs = actual_preview
        .iter()
        .zip(reference.waveform.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let stats = waveform_stats(waveform);
    assert!(
        stats.peak <= 1.1 && stats.clip_fraction < 0.001,
        "Rust audio codec clips Python eager codec groups: {stats:?}"
    );
    if let Some(expected) = reference.waveform_stats {
        assert!(
            (stats.rms - expected.rms).abs() <= 0.02
                && (stats.peak - expected.peak).abs() <= 0.2
                && preview_max_abs <= 0.2,
            "Rust audio codec differs from Python waveform: rust={stats:?}, python={expected:?}, preview_max_abs={preview_max_abs}"
        );
    }
}

fn first_mismatch_topk_report(
    actual: &[Vec<i32>],
    expected: &[Vec<i32>],
    all_step_topk: &[Vec<Vec<(usize, f32)>>],
) -> String {
    for (step_idx, (actual_step, expected_step)) in actual.iter().zip(expected.iter()).enumerate() {
        for (group_idx, (actual_id, expected_id)) in
            actual_step.iter().zip(expected_step.iter()).enumerate()
        {
            if actual_id != expected_id {
                let predictor_idx = group_idx.saturating_sub(1);
                let topk = all_step_topk
                    .get(step_idx)
                    .and_then(|step| step.get(predictor_idx));
                return format!(
                    "step {step_idx} group {group_idx} rust={actual_id} python={expected_id} topk={topk:?}"
                );
            }
        }
    }
    "none".to_string()
}

fn top5_last_position(logits: Tensor<Backend, 3>) -> Vec<(usize, f32)> {
    let [_batch_size, seq_len, vocab_size] = logits.dims();
    let values = logits
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let start = (seq_len - 1) * vocab_size;
    top5(values[start..start + vocab_size].to_vec())
}

fn codec_group_mismatch_summary(actual: &[Vec<i32>], expected: &[Vec<i32>]) -> String {
    let mut mismatch_count = 0_usize;
    let mut first = None;
    for (step_idx, (actual_step, expected_step)) in actual.iter().zip(expected.iter()).enumerate() {
        for (group_idx, (actual_id, expected_id)) in
            actual_step.iter().zip(expected_step.iter()).enumerate()
        {
            if actual_id != expected_id {
                mismatch_count += 1;
                first.get_or_insert((step_idx, group_idx, *actual_id, *expected_id));
            }
        }
    }
    match first {
        Some((step, group, actual_id, expected_id)) => format!(
            "codec mismatches={mismatch_count}, first=step {step} group {group}: rust={actual_id}, python={expected_id}"
        ),
        None => "codec mismatches=0".to_string(),
    }
}

fn waveform_stats(waveform: Tensor<Backend, 3>) -> WaveStats {
    let values = waveform
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum_sq = 0.0_f64;
    let mut clipped = 0_usize;
    for value in &values {
        min = min.min(*value);
        max = max.max(*value);
        sum_sq += (*value as f64) * (*value as f64);
        if value.abs() >= 0.999 {
            clipped += 1;
        }
    }
    let peak = min.abs().max(max.abs());
    WaveStats {
        min,
        max,
        peak,
        rms: (sum_sq / values.len() as f64).sqrt() as f32,
        clip_fraction: clipped as f32 / values.len() as f32,
    }
}
