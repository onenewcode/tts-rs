mod common;

use std::collections::BTreeMap;
use std::process::Command;

use burn::backend::{Flex, flex::FlexDevice};
use burn::nn::RmsNorm;
use burn::tensor::{DType, Tensor, TensorData};
use serde::Deserialize;
use tts_rs_qwen_burn::{
    CustomVoiceBatch, CustomVoiceRequest, KeyValueCache, Qwen3TtsTextTokenizer, SamplingConfig,
    StoppingRules, TalkerGenerateInput, build_custom_voice_prefill_batch, generate_talker_tokens,
    load_custom_voice_generation_config, load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;
const REPORT_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct TensorReference {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct DecodeStepReference {
    generation_step: usize,
    activations: BTreeMap<String, TensorReference>,
}

#[derive(Debug, Deserialize)]
struct TalkerDecodeReference {
    text: String,
    language: String,
    speaker: String,
    base_token_ids: Vec<i32>,
    steps: Vec<DecodeStepReference>,
}

#[test]
#[ignore = "loads real talker weights and compares V9 talker decode activations"]
fn v9_talker_decode_activations_match_python_oracle() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_talker_decode.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_talker_decode.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--max-new-tokens",
            "7",
            "--steps",
            "0,1,2,3,4",
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python talker decode oracle");
    assert!(status.success(), "Python talker decode oracle failed");

    let reference: TalkerDecodeReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();

    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device).unwrap();
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();
    let generation_config = load_custom_voice_generation_config(&model_dir).unwrap();
    let frontend = build_custom_voice_prefill_batch(
        &tokenizer,
        &talker.config.talker_config,
        &talker,
        &CustomVoiceBatch::single(CustomVoiceRequest {
            text: reference.text.clone(),
            language: Some(reference.language.clone()),
            speaker: Some(reference.speaker.clone()),
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
            collect_step_diagnostics: true,
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

    for step in &reference.steps {
        let actual = generated
            .step_diagnostics
            .get(step.generation_step)
            .unwrap_or_else(|| {
                panic!(
                    "missing Rust decode diagnostic step {}",
                    step.generation_step
                )
            });
        let mut summaries = Vec::new();
        for (name, expected) in &step.activations {
            let Some(actual_tensor) = actual.activations.get(name) else {
                continue;
            };
            summaries.push(compare_tensor(name, actual_tensor.clone(), expected));
        }
        let first_report = {
            let mut ordered = summaries.iter().collect::<Vec<_>>();
            ordered.sort_by_key(|summary| activation_order_key(&summary.name));
            ordered
                .into_iter()
                .filter(|summary| summary.exceed_count > 0)
                .take(16)
                .map(ActivationSummary::to_report_line)
                .collect::<Vec<_>>()
                .join("\n")
        };
        summaries.sort_by(|left, right| right.max_abs.total_cmp(&left.max_abs));
        let report = summaries
            .iter()
            .filter(|summary| summary.exceed_count > 0)
            .take(16)
            .map(ActivationSummary::to_report_line)
            .collect::<Vec<_>>()
            .join("\n");
        if report.is_empty() {
            println!(
                "V9 talker decode step {} activations match within {REPORT_TOLERANCE}",
                step.generation_step
            );
        } else {
            println!(
                "V9 talker decode step {} first mismatches:\n{first_report}\nV9 talker decode step {} top mismatches:\n{report}",
                step.generation_step, step.generation_step
            );
        }

        if step.generation_step == 0 {
            if let Some(probe) = probe_step0_layer0_post_norm_and_mlp(&talker, step, &device) {
                println!("{probe}");
            }
        }
    }
}

struct ActivationSummary {
    name: String,
    max_abs: f32,
    max_idx: usize,
    exceed_count: usize,
    actual_value: f32,
    expected_value: f32,
}

impl ActivationSummary {
    fn to_report_line(&self) -> String {
        format!(
            "{}: max_abs={} at {}, exceed_count={}, rust={}, python={}",
            self.name,
            self.max_abs,
            self.max_idx,
            self.exceed_count,
            self.actual_value,
            self.expected_value
        )
    }
}

fn compare_tensor(
    name: &str,
    actual: Tensor<Backend, 3>,
    expected: &TensorReference,
) -> ActivationSummary {
    if actual.dims().as_slice() != expected.shape.as_slice() {
        panic!(
            "{name}: shape mismatch rust={:?} python={:?}",
            actual.dims(),
            expected.shape
        );
    }
    let actual_values = actual
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let mut max_abs = 0.0_f32;
    let mut max_idx = 0_usize;
    let mut exceed_count = 0_usize;
    for (idx, (actual, expected)) in actual_values.iter().zip(expected.values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = idx;
        }
        if diff > REPORT_TOLERANCE {
            exceed_count += 1;
        }
    }
    ActivationSummary {
        name: name.to_string(),
        max_abs,
        max_idx,
        exceed_count,
        actual_value: actual_values[max_idx],
        expected_value: expected.values[max_idx],
    }
}

fn probe_step0_layer0_post_norm_and_mlp(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    step: &DecodeStepReference,
    device: &FlexDevice,
) -> Option<String> {
    let layer = talker.model.talker.model.layers.first()?;
    let decode_inputs_ref = step.activations.get("decode.inputs_embeds")?;
    let attn_residual_ref = step.activations.get("layers.0.attn_residual.output")?;
    let attn_output_ref = step.activations.get("layers.0.attn.output")?;
    let post_attention_norm_ref = step
        .activations
        .get("layers.0.post_attention_norm.output")?;
    let gate_ref = step.activations.get("layers.0.mlp.gate")?;
    let mlp_output_ref = step.activations.get("layers.0.mlp.output")?;
    let hidden_ref = step.activations.get("layers.0.hidden.output")?;
    let up_ref = step.activations.get("layers.0.mlp.up")?;

    let decode_inputs = tensor3_from_reference(decode_inputs_ref, device).cast(DType::BF16);
    let attn_residual = tensor3_from_reference(attn_residual_ref, device).cast(DType::BF16);
    let attn_output = tensor3_from_reference(attn_output_ref, device).cast(DType::BF16);
    let post_attention_norm =
        tensor3_from_reference(post_attention_norm_ref, device).cast(DType::BF16);
    let mlp_output = tensor3_from_reference(mlp_output_ref, device).cast(DType::BF16);

    let attn_residual_from_python_add = decode_inputs + attn_output;
    let attn_residual_add_summary = compare_tensor(
        "probe.layers.0.attn_residual_from_python_inputs_and_attn",
        attn_residual_from_python_add,
        attn_residual_ref,
    );

    let rust_post_norm = qwen_rms_norm_probe(&layer.post_attention_layernorm, attn_residual);
    let post_norm_summary = compare_tensor(
        "probe.layers.0.post_attention_norm_from_python_attn_residual",
        rust_post_norm,
        post_attention_norm_ref,
    );

    let rust_gate = layer.mlp.gate_proj.forward(post_attention_norm.clone());
    let gate_summary = compare_tensor(
        "probe.layers.0.mlp.gate_from_python_post_norm",
        rust_gate,
        gate_ref,
    );

    let rust_up = layer.mlp.up_proj.forward(post_attention_norm);
    let up_summary = compare_tensor(
        "probe.layers.0.mlp.up_from_python_post_norm",
        rust_up,
        up_ref,
    );

    let hidden_from_python_add =
        tensor3_from_reference(attn_residual_ref, device).cast(DType::BF16) + mlp_output;
    let hidden_add_summary = compare_tensor(
        "probe.layers.0.hidden_from_python_attn_residual_and_mlp_output",
        hidden_from_python_add,
        hidden_ref,
    );

    Some(
        [
            attn_residual_add_summary,
            post_norm_summary,
            gate_summary,
            up_summary,
            hidden_add_summary,
        ]
        .into_iter()
        .map(|summary| summary.to_report_line())
        .collect::<Vec<_>>()
        .join("\n"),
    )
}

fn tensor3_from_reference(reference: &TensorReference, device: &FlexDevice) -> Tensor<Backend, 3> {
    Tensor::<Backend, 3>::from_data(
        TensorData::new(
            reference.values.clone(),
            [reference.shape[0], reference.shape[1], reference.shape[2]],
        ),
        device,
    )
}

fn qwen_rms_norm_probe(norm: &RmsNorm<Backend>, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    let dtype = x.dtype();
    let x_f32 = x.cast(DType::F32);
    let variance = x_f32.clone().square().mean_dim(2);
    let normalized = x_f32 * (variance + norm.epsilon).sqrt().recip();
    normalized.cast(dtype) * norm.gamma.val().cast(dtype).unsqueeze()
}

fn activation_order_key(name: &str) -> (usize, usize, String) {
    if name == "decode.inputs_embeds" {
        return (0, 0, name.to_string());
    }
    let Some(rest) = name.strip_prefix("layers.") else {
        return (usize::MAX, usize::MAX, name.to_string());
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return (usize::MAX, usize::MAX, name.to_string());
    };
    let layer = layer.parse::<usize>().unwrap_or(usize::MAX);
    let stage = match suffix {
        "input_norm.output" => 0,
        "q_proj.output" => 1,
        "k_proj.output" => 2,
        "v_proj.output" => 3,
        "q_norm.output" => 4,
        "k_norm.output" => 5,
        "q_rot.output" => 6,
        "k_rot.output" => 7,
        "attn.weights" => 8,
        "attn.output" => 9,
        "attn_residual.output" => 10,
        "post_attention_norm.output" => 11,
        "mlp.gate" => 12,
        "mlp.up" => 13,
        "mlp.activated_gate" => 14,
        "mlp.product" => 15,
        "mlp.output" => 16,
        "hidden.output" => 17,
        _ => usize::MAX,
    };
    (layer + 1, stage, suffix.to_string())
}
