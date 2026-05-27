mod common;

use std::collections::BTreeMap;
use std::process::Command;

use burn::backend::{Flex, flex::FlexDevice};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Linear, RmsNorm};
use burn::tensor::activation::silu;
use burn::tensor::activation::softmax;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use tts_rs_qwen_burn::{
    CodePredictorGenerateInput, CodePredictorTeacherForcedInput, KeyValueCache, SamplingConfig,
    forward_code_predictor_teacher_forced, generate_code_predictor_groups,
    load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;
const REPORT_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct TensorReference {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CodePredictorStepReference {
    step_idx: usize,
    base_token_id: i32,
    expected_codec_groups: Vec<i32>,
    talker_hidden: TensorReference,
    topk: Vec<TopKReference>,
    scores: Vec<TensorReference>,
    teacher_forced_topk: Vec<TopKReference>,
    teacher_forced_scores: Vec<TensorReference>,
    activations: BTreeMap<String, BTreeMap<String, TensorReference>>,
}

#[derive(Debug, Deserialize)]
struct TopKReference {
    ids: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CodePredictorReference {
    steps: Vec<CodePredictorStepReference>,
}

#[test]
#[ignore = "loads real talker weights and checks code predictor with Python talker hidden states"]
fn v9_code_predictor_matches_python_with_python_hidden() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_code_predictor.json");
    let steps = std::env::var("QWEN_TTS_CODE_PREDICTOR_STEPS")
        .unwrap_or_else(|_| "0,1,2,3,4,5".to_string());
    let mut command = Command::new("uv");
    command
        .args([
            "run",
            "python",
            "py/generate_reference_v9_code_predictor.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--max-new-tokens",
            "7",
            "--steps",
            &steps,
            "--attention-implementation",
            "eager",
        ])
        .current_dir(common::workspace_root());
    if let Ok(text) = std::env::var("QWEN_TTS_TEXT") {
        command.args(["--text", &text]);
    }
    let status = command
        .status()
        .expect("failed to invoke Python code predictor oracle");
    assert!(status.success(), "Python code predictor oracle failed");

    let reference: CodePredictorReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();

    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device).unwrap();
    let cfg = &talker.config.talker_config;
    let mut mismatches = Vec::new();
    let collect_step_diagnostics =
        std::env::var("QWEN_TTS_CODE_PREDICTOR_FULL_RECOMPUTE").as_deref() != Ok("1");

    for step in &reference.steps {
        let hidden = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                step.talker_hidden.values.clone(),
                [step.talker_hidden.shape[0], step.talker_hidden.shape[1]],
            ),
            &device,
        )
        .cast(DType::BF16);
        let base_token = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(vec![step.base_token_id], [1, 1]),
            &device,
        );
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
                talker_hidden_state: hidden.clone(),
                base_codec_token_id: base_token.clone(),
                sampling: SamplingConfig::greedy(),
                collect_step_diagnostics,
            },
            &mut predictor_cache,
        )
        .unwrap();
        let actual = groups
            .codec_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        if actual != step.expected_codec_groups {
            let logit_summaries = groups
                .step_logits
                .iter()
                .zip(step.scores.iter())
                .enumerate()
                .map(|(idx, (actual_logits, expected_logits))| {
                    let summary = compare_last_logits(actual_logits.clone(), expected_logits);
                    format!(
                        "head {idx}: max_abs={} at {}, rust={}, python={}",
                        summary.max_abs,
                        summary.max_idx,
                        summary.actual_value,
                        summary.expected_value
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            let rust_topk = groups
                .step_logits
                .iter()
                .map(|logits| top5_last_position(logits.clone()))
                .collect::<Vec<_>>();
            mismatches.push(format!(
                "step {}: rust={:?}, python={:?}\n  rust_topk={:?}\n  python_topk={}",
                step.step_idx,
                actual,
                step.expected_codec_groups,
                rust_topk,
                format_python_topk(&step.topk)
            ));
            mismatches.push(format!("  logit_diff={logit_summaries}"));
            let activation_summaries = if collect_step_diagnostics {
                groups
                .step_diagnostics
                .iter()
                .enumerate()
                .filter_map(|(head_idx, diagnostic)| {
                    let expected = step.activations.get(&head_idx.to_string())?;
                    let mut summaries = expected
                        .iter()
                        .filter_map(|(name, expected_tensor)| {
                            if let Some(actual_tensor) = diagnostic.activations.get(name) {
                                Some(compare_activation(
                                    name,
                                    actual_tensor.clone(),
                                    expected_tensor,
                                ))
                            } else {
                                diagnostic.attention_activations.get(name).map(|actual_tensor| {
                                    compare_attention_activation(
                                        name,
                                        actual_tensor.clone(),
                                        expected_tensor,
                                    )
                                })
                            }
                        })
                        .collect::<Vec<_>>();
                    let first_report = {
                        let mut ordered = summaries.iter().collect::<Vec<_>>();
                        ordered.sort_by_key(|summary| activation_order_key(&summary.name));
                        ordered
                            .into_iter()
                            .filter(|summary| summary.exceed_count > 0)
                            .take(8)
                            .map(ActivationSummary::to_report_line)
                            .collect::<Vec<_>>()
                            .join("; ")
                    };
                    summaries.sort_by(|left, right| right.max_abs.total_cmp(&left.max_abs));
                    let report = summaries
                        .iter()
                        .filter(|summary| summary.exceed_count > 0)
                        .take(8)
                        .map(ActivationSummary::to_report_line)
                        .collect::<Vec<_>>()
                        .join("; ");
                    (!report.is_empty()).then(|| {
                        format!(
                            "  head {head_idx} first activations: {first_report}\n  head {head_idx} top activations: {report}"
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
            } else {
                String::new()
            };
            if !activation_summaries.is_empty() {
                mismatches.push(activation_summaries);
            }
            let head_from_python_hidden = groups
                .step_logits
                .iter()
                .enumerate()
                .filter_map(|(head_idx, _)| {
                    let expected_head = step.activations.get(&head_idx.to_string())?;
                    let expected_hidden = expected_head.get("model.norm.output")?;
                    let hidden = Tensor::<Backend, 3>::from_data(
                        TensorData::new(
                            expected_hidden.values.clone(),
                            [
                                expected_hidden.shape[0],
                                expected_hidden.shape[1],
                                expected_hidden.shape[2],
                            ],
                        ),
                        &device,
                    )
                    .cast(DType::BF16);
                    let [_batch_size, seq_len, hidden_size] = hidden.dims();
                    let hidden_2d = hidden.reshape([seq_len, hidden_size]);
                    let logits = talker.model.talker.code_predictor.lm_head[head_idx]
                        .forward(hidden_2d.clone())
                        .unsqueeze::<3>();
                    let summary = compare_last_logits(logits.clone(), &step.scores[head_idx]);
                    (summary.max_abs > REPORT_TOLERANCE).then(|| {
                        format!(
                            "  head {head_idx} logits_from_python_hidden: max_abs={} at {}, rust={}, python={}, rust_topk={:?}, python_topk={}",
                            summary.max_abs,
                            summary.max_idx,
                            summary.actual_value,
                            summary.expected_value,
                            top5_last_position(logits),
                            format_python_topk(&step.topk[head_idx..head_idx + 1])
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !head_from_python_hidden.is_empty() {
                mismatches.push(head_from_python_hidden);
            }
            if let Some(teacher_forced_report) =
                teacher_forced_probe(&talker, cfg, step, hidden.clone(), &device)
            {
                mismatches.push(teacher_forced_report);
            }
            if let Some(f32_report) = f32_lm_head_probe(&talker, &groups, step) {
                mismatches.push(f32_report);
            }
            if step.step_idx == 0 {
                if let Some(probe) =
                    probe_code_predictor_cache(step, 14, "step0.head14.cache", &predictor_cache)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 0, "step0.head0", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 12, "step0.head12", &device)
                {
                    mismatches.push(probe);
                }
                for layer_idx in 0..cfg.code_predictor_config.num_hidden_layers {
                    if let Some(probe) = probe_code_predictor_attention_from_python_cache(
                        &talker,
                        cfg,
                        step,
                        12,
                        layer_idx,
                        &format!("step0.head12.layer{layer_idx}"),
                        &device,
                    ) {
                        mismatches.push(probe);
                    }
                }
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 14, "step0.head14", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) = probe_code_predictor_head_layers(
                    &talker,
                    cfg,
                    step,
                    14,
                    "step0.head14.deep",
                    &[1, 2, 3, 4],
                    &device,
                ) {
                    mismatches.push(probe);
                }
            }
            if step.step_idx == 1 {
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 0, "step1.head0", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) = probe_step1_head0_o_proj_from_python_v0(&talker, step, &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) =
                    probe_step1_head0_attention_from_python_qkv(&talker, step, &device)
                {
                    mismatches.push(probe);
                }
            }
            if step.step_idx == 2 {
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 2, "step2.head2", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) = probe_code_predictor_head_layers(
                    &talker,
                    cfg,
                    step,
                    2,
                    "step2.head2.deep",
                    &[1],
                    &device,
                ) {
                    mismatches.push(probe);
                }
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 5, "step2.head5", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(diagnostic) = groups.step_diagnostics.get(6) {
                    if let Some(probe) = probe_code_predictor_diagnostic_cache(
                        step,
                        6,
                        "step2.head6.cache",
                        diagnostic,
                    ) {
                        mismatches.push(probe);
                    }
                }
                for layer_idx in 0..cfg.code_predictor_config.num_hidden_layers {
                    if let Some(probe) = probe_code_predictor_attention_from_python_cache(
                        &talker,
                        cfg,
                        step,
                        6,
                        layer_idx,
                        &format!("step2.head6.layer{layer_idx}"),
                        &device,
                    ) {
                        mismatches.push(probe);
                    }
                }
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 11, "step2.head11", &device)
                {
                    mismatches.push(probe);
                }
                if let Some(probe) = probe_code_predictor_head_layers(
                    &talker,
                    cfg,
                    step,
                    11,
                    "step2.head11.deep",
                    &[1, 2, 3, 4],
                    &device,
                ) {
                    mismatches.push(probe);
                }
                for layer_idx in 0..cfg.code_predictor_config.num_hidden_layers {
                    if let Some(probe) = probe_code_predictor_attention_from_python_cache(
                        &talker,
                        cfg,
                        step,
                        11,
                        layer_idx,
                        &format!("step2.head11.layer{layer_idx}"),
                        &device,
                    ) {
                        mismatches.push(probe);
                    }
                }
            }
            if step.step_idx == 3 {
                if let Some(probe) =
                    probe_code_predictor_head_layer0(&talker, step, 3, "step3.head3", &device)
                {
                    mismatches.push(probe);
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "code predictor mismatches with Python hidden states:\n{}",
        mismatches.join("\n")
    );
}

fn probe_code_predictor_diagnostic_cache(
    step: &CodePredictorStepReference,
    head_idx: usize,
    label: &str,
    diagnostic: &tts_rs_qwen_burn::CodePredictorGenerateStepDiagnostic<Backend>,
) -> Option<String> {
    let head = step.activations.get(&head_idx.to_string())?;
    let mut lines = vec![format!("  diagnostic cache probes {label}:")];
    for (name, actual) in &diagnostic.cache_activations {
        let Some(expected) = head.get(name) else {
            continue;
        };
        let summary = compare_attention_activation(
            &format!("probe.{label}.{name}"),
            actual.clone(),
            expected,
        );
        lines.push(format!("  {}", summary.to_report_line()));
        if name == "layers.1.cache.key" || name == "layers.1.cache.value" {
            lines.push(cache_position_summary(
                &format!("probe.{label}.{name}.by_position"),
                actual.clone(),
                expected,
            ));
        }
    }
    (lines.len() > 1).then(|| lines.join("\n"))
}

fn cache_position_summary(
    name: &str,
    actual: Tensor<Backend, 4>,
    expected: &TensorReference,
) -> String {
    let dims = actual.dims();
    if dims.as_slice() != expected.shape.as_slice() {
        return format!(
            "  {name}: shape mismatch rust={:?} python={:?}",
            dims, expected.shape
        );
    }
    let [_batch_size, num_heads, seq_len, head_dim] = dims;
    let actual_values = actual
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let mut per_pos = Vec::with_capacity(seq_len);
    for seq_idx in 0..seq_len {
        let mut max_abs = 0.0_f32;
        let mut max_head = 0_usize;
        let mut max_dim = 0_usize;
        let mut actual_value = 0.0_f32;
        let mut expected_value = 0.0_f32;
        let mut exceed_count = 0_usize;
        for head_idx in 0..num_heads {
            for dim_idx in 0..head_dim {
                let idx = (head_idx * seq_len + seq_idx) * head_dim + dim_idx;
                let diff = (actual_values[idx] - expected.values[idx]).abs();
                if diff > max_abs {
                    max_abs = diff;
                    max_head = head_idx;
                    max_dim = dim_idx;
                    actual_value = actual_values[idx];
                    expected_value = expected.values[idx];
                }
                if diff > REPORT_TOLERANCE {
                    exceed_count += 1;
                }
            }
        }
        per_pos.push(format!(
            "pos{seq_idx}: max_abs={max_abs} head={max_head} dim={max_dim} exceed_count={exceed_count} rust={actual_value} python={expected_value}"
        ));
    }
    format!("  {name}: {}", per_pos.join("; "))
}

fn f32_lm_head_probe(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    groups: &tts_rs_qwen_burn::CodePredictorGenerateOutput<Backend>,
    step: &CodePredictorStepReference,
) -> Option<String> {
    let actual = groups
        .codec_ids
        .clone()
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .ok()?;
    let first_mismatch = actual
        .iter()
        .zip(step.expected_codec_groups.iter())
        .position(|(left, right)| left != right)?;
    if first_mismatch == 0 {
        return None;
    }
    let head_idx = first_mismatch - 1;
    let diagnostic = groups.step_diagnostics.get(head_idx)?;
    let hidden = diagnostic.activations.get("model.norm.output")?.clone();
    let logits = linear_3d_f32_probe(
        &talker.model.talker.code_predictor.lm_head[head_idx],
        hidden,
    );
    Some(format!(
        "  f32_lm_head_probe head {head_idx}: rust_f32_topk={:?}",
        top5_last_position(logits)
    ))
}

fn linear_3d_f32_probe(linear: &Linear<Backend>, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    let [batch_size, seq_len, in_features] = x.dims();
    let out_features = linear.weight.dims()[1];
    let x_2d = x
        .reshape([batch_size * seq_len, in_features])
        .cast(DType::F32);
    let weight = linear.weight.val().cast(DType::F32);
    let output = match &linear.bias {
        Some(bias) => x_2d.matmul(weight) + bias.val().cast(DType::F32).unsqueeze::<2>(),
        None => x_2d.matmul(weight),
    };
    output.reshape([batch_size, seq_len, out_features])
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
            "{} max_abs={} at {}, exceed_count={}, rust={}, python={}",
            self.name,
            self.max_abs,
            self.max_idx,
            self.exceed_count,
            self.actual_value,
            self.expected_value
        )
    }
}

fn compare_activation(
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

fn compare_attention_activation(
    name: &str,
    actual: Tensor<Backend, 4>,
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

fn probe_code_predictor_head_layer0(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    step: &CodePredictorStepReference,
    head_idx: usize,
    label: &str,
    device: &FlexDevice,
) -> Option<String> {
    let head = step.activations.get(&head_idx.to_string())?;
    let layer = &talker.model.talker.code_predictor.model.layers[0];
    let expected_attn_residual = head.get("layers.0.attn_residual.output")?;
    let attn_residual = tensor3_from_reference(expected_attn_residual, device).cast(DType::BF16);
    let attn_residual_reload_summary = compare_activation(
        &format!("probe.{label}.layer0.attn_residual_json_reload"),
        attn_residual.clone(),
        expected_attn_residual,
    );
    let expected_post_norm = head.get("layers.0.post_attention_norm.output")?;
    let post_norm_from_python_residual =
        rms_norm_3d(&layer.post_attention_layernorm, attn_residual.clone());
    let post_norm_summary = compare_activation(
        &format!("probe.{label}.layer0.post_norm_from_python_attn_residual"),
        post_norm_from_python_residual,
        expected_post_norm,
    );
    let rms_variant_summaries = rms_norm_variant_summaries(
        &format!("probe.{label}.layer0.post_norm_variant"),
        &layer.post_attention_layernorm,
        attn_residual,
        expected_post_norm,
    );

    let post_norm = tensor3_from_reference(expected_post_norm, device).cast(DType::BF16);
    let expected_gate = head.get("layers.0.mlp.gate")?;
    let expected_up = head.get("layers.0.mlp.up")?;
    let gate_summary = compare_activation(
        &format!("probe.{label}.layer0.gate_from_python_post_norm"),
        layer.mlp.gate_proj.forward(post_norm.clone()),
        expected_gate,
    );
    let up_summary = compare_activation(
        &format!("probe.{label}.layer0.up_from_python_post_norm"),
        layer.mlp.up_proj.forward(post_norm),
        expected_up,
    );
    Some(format!(
        "  targeted probes:\n  {}\n  {}\n{}\n  {}\n  {}",
        attn_residual_reload_summary.to_report_line(),
        post_norm_summary.to_report_line(),
        rms_variant_summaries,
        gate_summary.to_report_line(),
        up_summary.to_report_line()
    ))
}

fn probe_code_predictor_cache(
    step: &CodePredictorStepReference,
    head_idx: usize,
    label: &str,
    cache: &[KeyValueCache<Backend>],
) -> Option<String> {
    let head = step.activations.get(&head_idx.to_string())?;
    let mut lines = vec![format!("  cache probes {label}:")];
    for (layer_idx, layer_cache) in cache.iter().enumerate() {
        let key_name = format!("layers.{layer_idx}.cache.key");
        let value_name = format!("layers.{layer_idx}.cache.value");
        if let (Some(actual), Some(expected)) = (layer_cache.key_snapshot(), head.get(&key_name)) {
            let summary = compare_attention_activation(
                &format!("probe.{label}.layer{layer_idx}.key"),
                actual,
                expected,
            );
            lines.push(format!("  {}", summary.to_report_line()));
        }
        if let (Some(actual), Some(expected)) =
            (layer_cache.value_snapshot(), head.get(&value_name))
        {
            let summary = compare_attention_activation(
                &format!("probe.{label}.layer{layer_idx}.value"),
                actual,
                expected,
            );
            lines.push(format!("  {}", summary.to_report_line()));
        }
    }
    (lines.len() > 1).then(|| lines.join("\n"))
}

fn probe_code_predictor_head_layers(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    cfg: &tts_rs_qwen_burn::Qwen3TtsTalkerConfig,
    step: &CodePredictorStepReference,
    head_idx: usize,
    label: &str,
    layer_indices: &[usize],
    device: &FlexDevice,
) -> Option<String> {
    let head = step.activations.get(&head_idx.to_string())?;
    let predictor_cfg = &cfg.code_predictor_config;
    let mut lines = vec![format!("  deep local probes {label}:")];

    for &layer_idx in layer_indices {
        let layer = talker
            .model
            .talker
            .code_predictor
            .model
            .layers
            .get(layer_idx)?;
        let previous_hidden_name = format!("layers.{}.hidden.output", layer_idx - 1);
        let input_norm_name = format!("layers.{layer_idx}.input_norm.output");
        let q_proj_name = format!("layers.{layer_idx}.q_proj.output");
        let k_proj_name = format!("layers.{layer_idx}.k_proj.output");
        let v_proj_name = format!("layers.{layer_idx}.v_proj.output");
        let q_norm_name = format!("layers.{layer_idx}.q_norm.output");
        let k_norm_name = format!("layers.{layer_idx}.k_norm.output");
        let attn_residual_name = format!("layers.{layer_idx}.attn_residual.output");
        let post_norm_name = format!("layers.{layer_idx}.post_attention_norm.output");
        let gate_name = format!("layers.{layer_idx}.mlp.gate");
        let up_name = format!("layers.{layer_idx}.mlp.up");
        let activated_gate_name = format!("layers.{layer_idx}.mlp.activated_gate");
        let product_name = format!("layers.{layer_idx}.mlp.product");
        let mlp_output_name = format!("layers.{layer_idx}.mlp.output");
        let hidden_name = format!("layers.{layer_idx}.hidden.output");

        let previous_hidden =
            tensor3_from_reference(head.get(&previous_hidden_name)?, device).cast(DType::BF16);
        let expected_input_norm = head.get(&input_norm_name)?;
        let input_norm_from_python_hidden =
            rms_norm_3d(&layer.input_layernorm, previous_hidden.clone());
        let input_norm_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.input_norm_from_python_hidden"),
            input_norm_from_python_hidden,
            expected_input_norm,
        );

        let input_norm = tensor3_from_reference(expected_input_norm, device).cast(DType::BF16);
        let expected_q_proj = head.get(&q_proj_name)?;
        let expected_k_proj = head.get(&k_proj_name)?;
        let expected_v_proj = head.get(&v_proj_name)?;
        let q_proj = layer.self_attn.q_proj.forward(input_norm.clone());
        let k_proj = layer.self_attn.k_proj.forward(input_norm.clone());
        let v_proj = layer.self_attn.v_proj.forward(input_norm);
        let q_proj_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.q_proj_from_python_input_norm"),
            q_proj.clone(),
            expected_q_proj,
        );
        let k_proj_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.k_proj_from_python_input_norm"),
            k_proj.clone(),
            expected_k_proj,
        );
        let v_proj_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.v_proj_from_python_input_norm"),
            v_proj,
            expected_v_proj,
        );

        let [batch_size, seq_len, _] = q_proj.dims();
        let q_norm = rms_norm_4d(
            &layer.self_attn.q_norm,
            q_proj.reshape([
                batch_size,
                seq_len,
                predictor_cfg.num_attention_heads,
                predictor_cfg.head_dim,
            ]),
        )
        .swap_dims(1, 2)
        .clone()
        .swap_dims(1, 2)
        .reshape([
            batch_size,
            seq_len,
            predictor_cfg.num_attention_heads * predictor_cfg.head_dim,
        ]);
        let k_norm = rms_norm_4d(
            &layer.self_attn.k_norm,
            k_proj.reshape([
                batch_size,
                seq_len,
                predictor_cfg.num_key_value_heads,
                predictor_cfg.head_dim,
            ]),
        )
        .swap_dims(1, 2)
        .clone()
        .swap_dims(1, 2)
        .reshape([
            batch_size,
            seq_len,
            predictor_cfg.num_key_value_heads * predictor_cfg.head_dim,
        ]);
        let q_norm_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.q_norm_from_python_q_proj"),
            q_norm,
            head.get(&q_norm_name)?,
        );
        let k_norm_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.k_norm_from_python_k_proj"),
            k_norm,
            head.get(&k_norm_name)?,
        );

        let attn_residual =
            tensor3_from_reference(head.get(&attn_residual_name)?, device).cast(DType::BF16);
        let expected_post_norm = head.get(&post_norm_name)?;
        let post_norm_from_python_residual =
            rms_norm_3d(&layer.post_attention_layernorm, attn_residual.clone());
        let post_norm_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.post_norm_from_python_attn_residual"),
            post_norm_from_python_residual,
            expected_post_norm,
        );

        let post_norm = tensor3_from_reference(expected_post_norm, device).cast(DType::BF16);
        let gate = layer.mlp.gate_proj.forward(post_norm.clone());
        let up = layer.mlp.up_proj.forward(post_norm);
        let gate_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.gate_from_python_post_norm"),
            gate.clone(),
            head.get(&gate_name)?,
        );
        let up_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.up_from_python_post_norm"),
            up.clone(),
            head.get(&up_name)?,
        );
        let dtype = gate.dtype();
        let activated_gate = silu(gate.cast(DType::F32)).cast(dtype);
        let activated_gate_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.activated_gate_from_python_gate"),
            activated_gate.clone(),
            head.get(&activated_gate_name)?,
        );
        let product = activated_gate * up;
        let product_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.product_from_python_gate_up"),
            product.clone(),
            head.get(&product_name)?,
        );

        let product_from_python =
            tensor3_from_reference(head.get(&product_name)?, device).cast(DType::BF16);
        let mlp_output = layer.mlp.down_proj.forward(product_from_python);
        let mlp_output_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.down_from_python_product"),
            mlp_output.clone(),
            head.get(&mlp_output_name)?,
        );
        let hidden_from_python_parts = attn_residual + mlp_output;
        let hidden_summary = compare_activation(
            &format!("probe.{label}.layer{layer_idx}.hidden_from_python_residual_and_mlp"),
            hidden_from_python_parts,
            head.get(&hidden_name)?,
        );

        lines.extend(
            [
                input_norm_summary,
                q_proj_summary,
                k_proj_summary,
                v_proj_summary,
                q_norm_summary,
                k_norm_summary,
                post_norm_summary,
                gate_summary,
                up_summary,
                activated_gate_summary,
                product_summary,
                mlp_output_summary,
                hidden_summary,
            ]
            .into_iter()
            .map(|summary| format!("  {}", summary.to_report_line())),
        );
    }

    Some(lines.join("\n"))
}

fn probe_step1_head0_o_proj_from_python_v0(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    step: &CodePredictorStepReference,
    device: &FlexDevice,
) -> Option<String> {
    let head = step.activations.get("0")?;
    let expected_v = head.get("layers.0.v_proj.output")?;
    let expected_attn = head.get("layers.0.attn.output")?;
    let cfg = &talker.config.talker_config.code_predictor_config;
    let pre_o_hidden_size = cfg.num_attention_heads * cfg.head_dim;
    let output_hidden_size = expected_attn.shape[2];
    let kv_hidden_size = cfg.num_key_value_heads * cfg.head_dim;
    let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;

    let v = tensor3_from_reference(expected_v, device).cast(DType::BF16);
    let v0 = v.slice([0..1, 0..1, 0..kv_hidden_size]);
    let pre_o = v0
        .reshape([1, 1, cfg.num_key_value_heads, cfg.head_dim])
        .swap_dims(1, 2)
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, n_rep)
        .reshape([1, cfg.num_attention_heads, 1, cfg.head_dim])
        .swap_dims(1, 2)
        .clone()
        .reshape([1, 1, pre_o_hidden_size]);
    let actual = talker.model.talker.code_predictor.model.layers[0]
        .self_attn
        .o_proj
        .forward(pre_o.reshape([1, pre_o_hidden_size]))
        .unsqueeze::<3>();

    let expected = TensorReference {
        shape: vec![1, 1, output_hidden_size],
        values: expected_attn.values[0..output_hidden_size].to_vec(),
    };
    let summary = compare_activation(
        "probe.step1.head0.layer0.o_proj_from_python_v0",
        actual,
        &expected,
    );
    Some(format!(
        "  attention first-token probe:\n  {}",
        summary.to_report_line()
    ))
}

fn probe_step1_head0_attention_from_python_qkv(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    step: &CodePredictorStepReference,
    device: &FlexDevice,
) -> Option<String> {
    let head = step.activations.get("0")?;
    let expected_q =
        tensor3_from_reference(head.get("layers.0.q_rot.output")?, device).cast(DType::BF16);
    let expected_k =
        tensor3_from_reference(head.get("layers.0.k_rot.output")?, device).cast(DType::BF16);
    let expected_v =
        tensor3_from_reference(head.get("layers.0.v_proj.output")?, device).cast(DType::BF16);
    let expected_attn = head.get("layers.0.attn.output")?;
    let cfg = &talker.config.talker_config.code_predictor_config;
    let [batch_size, seq_len, _] = expected_q.dims();
    let pre_o_hidden_size = cfg.num_attention_heads * cfg.head_dim;
    let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;

    let q = expected_q
        .reshape([batch_size, seq_len, cfg.num_attention_heads, cfg.head_dim])
        .swap_dims(1, 2);
    let k = expected_k
        .reshape([batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        .swap_dims(1, 2);
    let v = expected_v
        .reshape([batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        .swap_dims(1, 2);
    let k = repeat_kv_for_probe(k, n_rep);
    let v = repeat_kv_for_probe(v, n_rep);
    let mask =
        generate_autoregressive_mask::<Backend>(batch_size, seq_len, device).unsqueeze_dim::<4>(1);
    let scores = q
        .cast(DType::F32)
        .matmul(k.cast(DType::F32).swap_dims(2, 3))
        .div_scalar((cfg.head_dim as f32).sqrt())
        .mask_fill(mask, f32::NEG_INFINITY);
    let weights = softmax(scores, 3);
    let pre_o = weights
        .matmul(v.cast(DType::F32))
        .cast(DType::BF16)
        .swap_dims(1, 2)
        .clone()
        .reshape([batch_size, seq_len, pre_o_hidden_size]);
    let actual = talker.model.talker.code_predictor.model.layers[0]
        .self_attn
        .o_proj
        .forward(pre_o.reshape([batch_size * seq_len, pre_o_hidden_size]))
        .reshape([batch_size, seq_len, expected_attn.shape[2]]);
    let summary = compare_activation(
        "probe.step1.head0.layer0.attn_from_python_qkv",
        actual,
        expected_attn,
    );
    Some(format!(
        "  attention qkv probe:\n  {}",
        summary.to_report_line()
    ))
}

fn probe_code_predictor_attention_from_python_cache(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    cfg: &tts_rs_qwen_burn::Qwen3TtsTalkerConfig,
    step: &CodePredictorStepReference,
    head_idx: usize,
    layer_idx: usize,
    label: &str,
    device: &FlexDevice,
) -> Option<String> {
    let head = step.activations.get(&head_idx.to_string())?;
    let layer = &talker.model.talker.code_predictor.model.layers[layer_idx];
    let predictor_cfg = &cfg.code_predictor_config;
    let expected_q = tensor3_from_reference(
        head.get(&format!("layers.{layer_idx}.q_rot.output"))?,
        device,
    )
    .cast(DType::BF16);
    let expected_key =
        tensor4_from_reference(head.get(&format!("layers.{layer_idx}.cache.key"))?, device)
            .cast(DType::BF16);
    let expected_value = tensor4_from_reference(
        head.get(&format!("layers.{layer_idx}.cache.value"))?,
        device,
    )
    .cast(DType::BF16);
    let expected_weights = tensor4_from_reference(
        head.get(&format!("layers.{layer_idx}.attn.weights"))?,
        device,
    )
    .cast(DType::BF16);
    let expected_weights_ref = head.get(&format!("layers.{layer_idx}.attn.weights"))?;
    let expected_scores_ref = head.get(&format!("layers.{layer_idx}.attn.scores"));
    let expected_manual_weights_ref = head.get(&format!("layers.{layer_idx}.attn.manual_weights"));
    let expected_attn = head.get(&format!("layers.{layer_idx}.attn.output"))?;

    let [batch_size, seq_len, _] = expected_q.dims();
    let pre_o_hidden_size = predictor_cfg.num_attention_heads * predictor_cfg.head_dim;
    let n_rep = predictor_cfg.num_attention_heads / predictor_cfg.num_key_value_heads;
    let q = expected_q
        .reshape([
            batch_size,
            seq_len,
            predictor_cfg.num_attention_heads,
            predictor_cfg.head_dim,
        ])
        .swap_dims(1, 2);
    let k = repeat_kv_for_probe(expected_key, n_rep);
    let v = repeat_kv_for_probe(expected_value, n_rep);
    let scaling = (predictor_cfg.head_dim as f32).sqrt().recip();

    let eager_bf16_scores = q
        .clone()
        .matmul(k.clone().swap_dims(2, 3).clone())
        .mul_scalar(scaling);
    let eager_bf16_scores_summary = expected_scores_ref.map(|expected_scores| {
        compare_attention_activation(
            &format!("probe.{label}.attention_scores_from_python_cache.eager_bf16_scores"),
            eager_bf16_scores.clone(),
            expected_scores,
        )
    });
    let eager_bf16_weights = softmax(eager_bf16_scores.cast(DType::F32), 3).cast(DType::BF16);
    let eager_bf16_weights_summary = compare_attention_activation(
        &format!("probe.{label}.attention_weights_from_python_cache.eager_bf16_scores"),
        eager_bf16_weights.clone(),
        expected_weights_ref,
    );
    let eager_bf16_output = attention_o_proj_from_weights(
        layer,
        eager_bf16_weights.matmul(v.clone()),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );
    let pytorch_bf16_scores = pytorch_bf16_scores_for_probe(q.clone(), k.clone(), scaling, device);
    let pytorch_bf16_scores_summary = expected_scores_ref.map(|expected_scores| {
        compare_attention_activation(
            &format!("probe.{label}.attention_scores_from_python_cache.pytorch_bf16_scores"),
            pytorch_bf16_scores.clone(),
            expected_scores,
        )
    });
    let pytorch_bf16_weights = softmax(pytorch_bf16_scores.cast(DType::F32), 3).cast(DType::BF16);
    let pytorch_bf16_weights_summary = compare_attention_activation(
        &format!("probe.{label}.attention_weights_from_python_cache.pytorch_bf16_scores"),
        pytorch_bf16_weights.clone(),
        expected_weights_ref,
    );
    let pytorch_bf16_output = attention_o_proj_from_weights(
        layer,
        pytorch_bf16_weights.matmul(v.clone()),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );

    let f32_scores = q
        .clone()
        .cast(DType::F32)
        .matmul(k.clone().cast(DType::F32).swap_dims(2, 3).clone())
        .mul_scalar(scaling);
    let f32_scores_summary = expected_scores_ref.map(|expected_scores| {
        compare_attention_activation(
            &format!("probe.{label}.attention_scores_from_python_cache.f32_scores"),
            f32_scores.clone(),
            expected_scores,
        )
    });
    let f32_weights = softmax(f32_scores, 3);
    let f32_cast_weights = f32_weights.clone().cast(DType::BF16);
    let f32_cast_weights_summary = compare_attention_activation(
        &format!("probe.{label}.attention_weights_from_python_cache.f32_scores"),
        f32_cast_weights.clone(),
        expected_weights_ref,
    );
    let cast_softmax_output = attention_o_proj_from_weights(
        layer,
        f32_cast_weights
            .clone()
            .cast(DType::F32)
            .matmul(v.clone().cast(DType::F32))
            .cast(DType::BF16),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );
    let keep_f32_output = attention_o_proj_from_weights(
        layer,
        f32_weights
            .matmul(v.clone().cast(DType::F32))
            .cast(DType::BF16),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );
    let python_weights_bf16_value_output = attention_o_proj_from_weights(
        layer,
        expected_weights.clone().matmul(v.clone()),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );
    let python_weights_f32_value_output = attention_o_proj_from_weights(
        layer,
        expected_weights
            .clone()
            .cast(DType::F32)
            .matmul(v.cast(DType::F32))
            .cast(DType::BF16),
        batch_size,
        seq_len,
        pre_o_hidden_size,
    );

    let python_manual_weights_summary =
        expected_manual_weights_ref.map(|expected_manual_weights| {
            compare_attention_activation(
                &format!("probe.{label}.python_captured_weights_vs_manual_weights"),
                expected_weights.clone(),
                expected_manual_weights,
            )
        });
    let mut lines = Vec::new();
    if let Some(summary) = eager_bf16_scores_summary {
        lines.push(format!("  {}", summary.to_report_line()));
    }
    if let Some(summary) = pytorch_bf16_scores_summary {
        lines.push(format!("  {}", summary.to_report_line()));
    }
    if let Some(summary) = f32_scores_summary {
        lines.push(format!("  {}", summary.to_report_line()));
    }
    lines.push(format!("  {}", eager_bf16_weights_summary.to_report_line()));
    lines.push(format!(
        "  {}",
        pytorch_bf16_weights_summary.to_report_line()
    ));
    lines.push(format!("  {}", f32_cast_weights_summary.to_report_line()));
    if let Some(summary) = python_manual_weights_summary {
        lines.push(format!("  {}", summary.to_report_line()));
    }
    lines.extend(
        [
            ("eager_bf16_scores_bf16_value", eager_bf16_output),
            ("pytorch_bf16_scores_bf16_value", pytorch_bf16_output),
            ("f32_scores_cast_softmax_f32_value", cast_softmax_output),
            ("f32_scores_f32_softmax_f32_value", keep_f32_output),
            (
                "python_weights_bf16_value",
                python_weights_bf16_value_output,
            ),
            ("python_weights_f32_value", python_weights_f32_value_output),
        ]
        .into_iter()
        .map(|(variant, actual)| {
            let summary = compare_activation(
                &format!("probe.{label}.attention_from_python_cache.{variant}"),
                actual,
                expected_attn,
            );
            format!("  {}", summary.to_report_line())
        }),
    );

    Some(format!(
        "  attention-cache probes {label}:\n{}",
        lines.join("\n")
    ))
}

fn attention_o_proj_from_weights(
    layer: &tts_rs_qwen_burn::talker::Qwen3TtsDecoderLayer<Backend>,
    pre_o: Tensor<Backend, 4>,
    batch_size: usize,
    seq_len: usize,
    pre_o_hidden_size: usize,
) -> Tensor<Backend, 3> {
    let pre_o = pre_o
        .swap_dims(1, 2)
        .clone()
        .reshape([batch_size, seq_len, pre_o_hidden_size]);
    layer
        .self_attn
        .o_proj
        .forward(pre_o.reshape([batch_size * seq_len, pre_o_hidden_size]))
        .reshape([batch_size, seq_len, layer.self_attn.o_proj.weight.dims()[1]])
}

fn pytorch_bf16_scores_for_probe(
    q: Tensor<Backend, 4>,
    k: Tensor<Backend, 4>,
    scaling: f32,
    device: &FlexDevice,
) -> Tensor<Backend, 4> {
    let [batch_size, num_heads, query_len, head_dim] = q.dims();
    let [_k_batch_size, _k_num_heads, key_len, _k_head_dim] = k.dims();
    let q_values = q
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("probe q values should be convertible to f32");
    let k_values = k
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("probe k values should be convertible to f32");
    let mut scores = Vec::with_capacity(batch_size * num_heads * query_len * key_len);

    for batch_idx in 0..batch_size {
        for head_idx in 0..num_heads {
            for query_idx in 0..query_len {
                let q_base =
                    ((batch_idx * num_heads + head_idx) * query_len + query_idx) * head_dim;
                for key_idx in 0..key_len {
                    let k_base =
                        ((batch_idx * num_heads + head_idx) * key_len + key_idx) * head_dim;
                    let mut sum = 0.0_f32;
                    for dim_idx in 0..head_dim {
                        sum += q_values[q_base + dim_idx] * k_values[k_base + dim_idx];
                    }
                    scores.push(round_f32_to_bf16_for_probe(
                        round_f32_to_bf16_for_probe(sum) * scaling,
                    ));
                }
            }
        }
    }

    Tensor::<Backend, 4>::from_data(
        TensorData::new(scores, [batch_size, num_heads, query_len, key_len]),
        device,
    )
    .cast(DType::BF16)
}

fn round_f32_to_bf16_for_probe(value: f32) -> f32 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn repeat_kv_for_probe(x: Tensor<Backend, 4>, n_rep: usize) -> Tensor<Backend, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
        batch_size,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    ])
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

fn tensor4_from_reference(reference: &TensorReference, device: &FlexDevice) -> Tensor<Backend, 4> {
    Tensor::<Backend, 4>::from_data(
        TensorData::new(
            reference.values.clone(),
            [
                reference.shape[0],
                reference.shape[1],
                reference.shape[2],
                reference.shape[3],
            ],
        ),
        device,
    )
}

fn teacher_forced_probe(
    talker: &tts_rs_qwen_burn::LoadedQwen3TtsTalker<Backend>,
    cfg: &tts_rs_qwen_burn::Qwen3TtsTalkerConfig,
    step: &CodePredictorStepReference,
    hidden: Tensor<Backend, 2>,
    device: &FlexDevice,
) -> Option<String> {
    if step.teacher_forced_scores.is_empty() {
        return None;
    }
    let codec_ids = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(
            step.expected_codec_groups.clone(),
            [1, step.expected_codec_groups.len()],
        ),
        device,
    );
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
    let output = forward_code_predictor_teacher_forced(
        cfg,
        talker,
        CodePredictorTeacherForcedInput {
            talker_hidden_states: hidden,
            codec_ids,
            attention_mask: None,
            collect_activations: false,
        },
        &mut predictor_cache,
    )
    .ok()?;

    let [_batch_size, num_heads, vocab_size] = output.logits.dims();
    let mut lines = Vec::new();
    for head_idx in 0..num_heads.min(step.teacher_forced_scores.len()) {
        let logits = output
            .logits
            .clone()
            .slice([0..1, head_idx..head_idx + 1, 0..vocab_size]);
        let summary = compare_last_logits(logits.clone(), &step.teacher_forced_scores[head_idx]);
        if summary.max_abs > REPORT_TOLERANCE {
            lines.push(format!(
                "head {head_idx}: max_abs={} at {}, rust={}, python={}, rust_topk={:?}, python_topk={}",
                summary.max_abs,
                summary.max_idx,
                summary.actual_value,
                summary.expected_value,
                top5_last_position(logits),
                format_python_topk(&step.teacher_forced_topk[head_idx..head_idx + 1])
            ));
        }
    }
    (!lines.is_empty()).then(|| format!("  teacher_forced_logits:\n  {}", lines.join("\n  ")))
}

fn rms_norm_3d(norm: &RmsNorm<Backend>, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    let dtype = x.dtype();
    let x = x.cast(DType::F32);
    let variance = x.clone().square().mean_dim(2);
    let x = x * (variance + norm.epsilon).sqrt().recip();
    norm.gamma.val().cast(dtype).unsqueeze() * cast_bf16_with_signed_tie_bias(x, 1.0e-6)
}

fn rms_norm_4d(norm: &RmsNorm<Backend>, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let dtype = x.dtype();
    let x = x.cast(DType::F32);
    let variance = x.clone().square().mean_dim(3);
    let x = x * (variance + norm.epsilon).sqrt().recip();
    norm.gamma.val().cast(dtype).unsqueeze() * cast_bf16_with_signed_tie_bias_4d(x, 1.0e-6)
}

fn rms_norm_variant_summaries(
    label: &str,
    norm: &RmsNorm<Backend>,
    x: Tensor<Backend, 3>,
    expected: &TensorReference,
) -> String {
    let dtype = x.dtype();
    let x_f32 = x.clone().cast(DType::F32);
    let variance = x_f32.clone().square().mean_dim(2);
    let denom = (variance.clone() + norm.epsilon).sqrt();
    let inv = denom.clone().recip();
    let normalized_f32 = x_f32.clone() * inv.clone();
    let normalized_bf16 = normalized_f32.clone().cast(dtype);
    let gamma_bf16 = norm.gamma.val().cast(dtype).unsqueeze();
    let gamma_f32 = norm.gamma.val().cast(DType::F32).unsqueeze();

    let variants = [
        (
            "custom_gamma_first",
            gamma_bf16.clone() * normalized_bf16.clone(),
        ),
        (
            "custom_x_first",
            normalized_bf16.clone() * gamma_bf16.clone(),
        ),
        (
            "divide_then_gamma",
            gamma_bf16.clone() * (x_f32.clone() / denom.clone()).cast(dtype),
        ),
        (
            "gamma_f32_cast_end",
            (gamma_f32.clone() * normalized_f32.clone()).cast(dtype),
        ),
        (
            "normalized_bf16_f32_mul_cast_end",
            (gamma_f32 * normalized_bf16.cast(DType::F32)).cast(dtype),
        ),
        (
            "pytorch_like_bf16_round",
            gamma_bf16.clone() * cast_bf16_round_to_nearest(normalized_f32.clone()),
        ),
        (
            "signed_tie_bias_1e_6",
            gamma_bf16.clone() * cast_bf16_with_signed_tie_bias(normalized_f32.clone(), 1.0e-6),
        ),
        (
            "signed_tie_bias_1e_7",
            gamma_bf16.clone() * cast_bf16_with_signed_tie_bias(normalized_f32.clone(), 1.0e-7),
        ),
        (
            "signed_tie_bias_1e_8",
            gamma_bf16.clone() * cast_bf16_with_signed_tie_bias(normalized_f32, 1.0e-8),
        ),
        ("burn_native", norm.forward(x)),
    ];

    variants
        .into_iter()
        .map(|(variant, actual)| {
            let summary = compare_activation(&format!("{label}.{variant}"), actual, expected);
            format!("  {}", summary.to_report_line())
        })
        .chain(std::iter::once(format!(
            "  {label}.variance_probe {}",
            rms_norm_variance_probe(x_f32, norm.epsilon)
        )))
        .collect::<Vec<_>>()
        .join("\n")
}

fn cast_bf16_round_to_nearest(x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    let abs = x.clone().abs();
    let exponent = (abs.clone() + f32::MIN_POSITIVE)
        .log()
        .div_scalar(std::f32::consts::LN_2)
        .floor();
    let half_ulp = abs.full_like(2.0).powf(exponent - 8.0);
    let signed_half_ulp = half_ulp
        .clone()
        .mask_where(x.clone().lower_elem(0.0), -half_ulp);
    (x + signed_half_ulp).cast(DType::BF16)
}

fn cast_bf16_with_signed_tie_bias(x: Tensor<Backend, 3>, eps: f64) -> Tensor<Backend, 3> {
    let positive = x.clone() + eps;
    let negative = x.clone() - eps;
    positive
        .mask_where(x.clone().lower_elem(0.0), negative)
        .cast(DType::BF16)
}

fn cast_bf16_with_signed_tie_bias_4d(x: Tensor<Backend, 4>, eps: f64) -> Tensor<Backend, 4> {
    let positive = x.clone() + eps;
    let negative = x.clone() - eps;
    positive
        .mask_where(x.clone().lower_elem(0.0), negative)
        .cast(DType::BF16)
}

fn rms_norm_variance_probe(x_f32: Tensor<Backend, 3>, epsilon: f64) -> String {
    let x_original = x_f32.clone();
    let variance = x_f32.square().mean_dim(2);
    let inv = (variance.clone() + epsilon).sqrt().recip();
    let normalized_f32 = x_original.clone() * inv.clone();
    let normalized_bf16 = normalized_f32.clone().cast(DType::BF16);
    let variance_values = variance
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let inv_values = inv.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let x_values = x_original
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let normalized_f32_values = normalized_f32
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let normalized_bf16_values = normalized_bf16
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let idx = 100.min(x_values.len().saturating_sub(1));
    format!(
        "variance={variance_values:?}, inv_rsqrt={inv_values:?}, idx{idx}: x={}, normed_f32={}, normed_bf16={}",
        x_values[idx], normalized_f32_values[idx], normalized_bf16_values[idx]
    )
}

fn activation_order_key(name: &str) -> (usize, usize, String) {
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
    (layer, stage, suffix.to_string())
}

struct LogitSummary {
    max_abs: f32,
    max_idx: usize,
    actual_value: f32,
    expected_value: f32,
}

fn compare_last_logits(actual: Tensor<Backend, 3>, expected: &TensorReference) -> LogitSummary {
    let [_batch_size, seq_len, vocab_size] = actual.dims();
    let values = actual
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let start = (seq_len - 1) * vocab_size;
    let actual_last = &values[start..start + vocab_size];
    let expected_values = &expected.values;
    let expected_last = if expected.shape.len() == 3 && expected.shape[1] > 1 {
        let expected_start = (expected.shape[1] - 1) * expected.shape[2];
        &expected_values[expected_start..expected_start + expected.shape[2]]
    } else {
        expected_values.as_slice()
    };
    let mut max_abs = 0.0_f32;
    let mut max_idx = 0_usize;
    for (idx, (actual, expected)) in actual_last.iter().zip(expected_last.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = idx;
        }
    }
    LogitSummary {
        max_abs,
        max_idx,
        actual_value: actual_last[max_idx],
        expected_value: expected_last[max_idx],
    }
}

fn top5_last_position(logits: Tensor<Backend, 3>) -> Vec<(usize, f32)> {
    let [_batch_size, seq_len, vocab_size] = logits.dims();
    let values = logits
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let start = (seq_len - 1) * vocab_size;
    let end = start + vocab_size;
    let mut indexed = values[start..end]
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(5);
    indexed
}

fn format_python_topk(topk: &[TopKReference]) -> String {
    topk.iter()
        .map(|group| {
            group
                .ids
                .iter()
                .copied()
                .zip(group.values.iter().copied())
                .collect::<Vec<_>>()
        })
        .map(|items| format!("{items:?}"))
        .collect::<Vec<_>>()
        .join("; ")
}
