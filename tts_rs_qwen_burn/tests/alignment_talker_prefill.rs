mod common;

use std::collections::BTreeMap;
use std::process::Command;

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use tts_rs_qwen_burn::{
    KeyValueCache, TalkerForwardInput, forward_talker_prefill, load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;
const REPORT_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct TensorReference {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TalkerPrefillReference {
    inputs_embeds: TensorReference,
    position_ids: Vec<Vec<Vec<i32>>>,
    attention_mask: Vec<Vec<i32>>,
    activations: BTreeMap<String, TensorReference>,
}

#[test]
#[ignore = "loads real talker weights and emits detailed V9 prefill activations"]
fn v9_talker_prefill_activations_match_python_oracle() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_talker_prefill.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_talker_prefill.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--max-layers",
            "4",
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python talker prefill oracle");
    assert!(status.success(), "Python talker prefill oracle failed");

    let reference: TalkerPrefillReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();

    let device = Default::default();
    let loaded = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device)
        .expect("talker should load");
    let cfg = &loaded.config.talker_config;
    let mut cache = (0..cfg.num_hidden_layers)
        .map(|_| KeyValueCache::new(1, cfg.num_key_value_heads, 4096, cfg.head_dim))
        .collect::<Vec<_>>();

    let inputs_embeds = tensor3_from_reference(&reference.inputs_embeds, &device).cast(DType::BF16);
    let position_ids = Tensor::<Backend, 3, Int>::from_data(
        TensorData::new(
            reference
                .position_ids
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            dims3_i32(&reference.position_ids),
        ),
        &device,
    );
    let attention_mask = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(
            reference
                .attention_mask
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            dims2_i32(&reference.attention_mask),
        ),
        &device,
    );

    let output = forward_talker_prefill(
        cfg,
        &loaded,
        TalkerForwardInput {
            inputs_embeds,
            position_ids,
            attention_mask: Some(attention_mask),
            collect_activations: true,
        },
        &mut cache,
    )
    .expect("talker prefill should run");

    let mut summaries = Vec::new();
    for (name, expected) in reference.activations.iter() {
        let actual = output
            .activations
            .get(name)
            .unwrap_or_else(|| panic!("missing Rust activation {name}"));
        summaries.push(compare_tensor(name, actual.clone(), expected));
    }
    if let (Some(mlp_input), Some(gate_expected)) = (
        reference
            .activations
            .get("layers.0.post_attention_norm.output"),
        reference.activations.get("layers.0.mlp.gate"),
    ) {
        let gate_from_python_input = loaded.model.talker.model.layers[0]
            .mlp
            .gate_proj
            .forward(tensor3_from_reference(mlp_input, &device).cast(DType::BF16));
        let gate_summary = compare_tensor(
            "probe.layers.0.mlp.gate_from_python_input",
            gate_from_python_input,
            gate_expected,
        );
        println!("{}", gate_summary.to_report_line());
    }
    let ordered_summaries = summaries
        .iter()
        .filter(|summary| summary.exceed_count > 0)
        .map(ActivationSummary::to_report_line)
        .take(24)
        .collect::<Vec<_>>()
        .join("\n");
    summaries.sort_by(|left, right| right.max_abs.total_cmp(&left.max_abs));
    let report = summaries
        .iter()
        .filter(|summary| summary.exceed_count > 0)
        .take(12)
        .map(ActivationSummary::to_report_line)
        .collect::<Vec<_>>()
        .join("\n");
    if report.is_empty() {
        println!("V9 talker prefill activations match within {REPORT_TOLERANCE}");
    } else {
        println!("V9 talker prefill top mismatches:\n{report}");
        println!("V9 talker prefill first mismatches by oracle order:\n{ordered_summaries}");
        let layer0_report = summaries
            .iter()
            .filter(|summary| summary.name.starts_with("layers.0."))
            .map(ActivationSummary::to_report_line)
            .collect::<Vec<_>>()
            .join("\n");
        println!("V9 talker prefill layer0 summary:\n{layer0_report}");
    }
    if std::env::var_os("QWEN_TTS_STRICT_TALKER_PREFILL_ALIGNMENT").is_some() {
        assert!(
            summaries.iter().all(|summary| summary.exceed_count == 0),
            "V9 talker prefill strict alignment failed:\n{report}"
        );
    }
}

fn tensor3_from_reference<B: burn::tensor::backend::Backend>(
    reference: &TensorReference,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(
            reference.values.clone(),
            [reference.shape[0], reference.shape[1], reference.shape[2]],
        ),
        device,
    )
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
    if actual_values.len() != expected.values.len() {
        panic!(
            "{name}: len mismatch rust={} python={}",
            actual_values.len(),
            expected.values.len()
        );
    }
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

fn dims3_i32(values: &[Vec<Vec<i32>>]) -> [usize; 3] {
    [
        values.len(),
        values.first().map_or(0, Vec::len),
        values
            .first()
            .and_then(|batch| batch.first())
            .map_or(0, Vec::len),
    ]
}

fn dims2_i32(values: &[Vec<i32>]) -> [usize; 2] {
    [values.len(), values.first().map_or(0, Vec::len)]
}
