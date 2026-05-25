use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use tts_rs_qwen_burn::{
    KeyValueCache, TalkerForwardInput, forward_talker_prefill, load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;

#[derive(Deserialize)]
struct ReferenceData {
    input: InputData,
    expected: ExpectedData,
}

#[derive(Deserialize)]
struct InputData {
    inputs_embeds: Vec<Vec<Vec<f32>>>,
    position_ids: Vec<Vec<Vec<i32>>>,
}

#[derive(Deserialize)]
struct ExpectedData {
    logits: TensorStats,
    layer_0_output: TensorStats,
    final_norm: TensorStats,
    #[serde(flatten)]
    activations: BTreeMap<String, TensorStats>,
}

#[derive(Deserialize)]
struct TensorStats {
    shape: Vec<usize>,
    sum: f32,
    first_5: Vec<f32>,
    values: Option<Vec<f32>>,
}

#[test]
#[ignore] // Run manually with `cargo test --test talker_alignment -- --ignored`
fn test_numerical_alignment_with_python_reference() {
    let device = Default::default();

    // 1. Load reference data
    let ref_path = "../reference.json";
    let ref_json = fs::read_to_string(ref_path)
        .expect("reference.json not found. Run `python py/generate_reference.py` first.");
    let ref_data: ReferenceData = serde_json::from_str(&ref_json).unwrap();

    // 2. Load Rust Model
    let model_dir = "../Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
    let loaded = load_qwen3_tts_talker_for_inference::<Backend>(model_dir, &device)
        .expect("Failed to load Rust model");

    // 3. Prepare Input Tensors from JSON
    let flattened_embeds: Vec<f32> = ref_data
        .input
        .inputs_embeds
        .iter()
        .flatten()
        .flatten()
        .cloned()
        .collect();
    let inputs_embeds =
        Tensor::<Backend, 3>::from_data(TensorData::new(flattened_embeds, [1, 5, 1024]), &device)
            .cast(DType::BF16); // Match model weights

    let flattened_pos: Vec<i32> = ref_data
        .input
        .position_ids
        .iter()
        .flatten()
        .flatten()
        .cloned()
        .collect();
    let position_ids =
        Tensor::<Backend, 3, Int>::from_data(TensorData::new(flattened_pos, [3, 1, 5]), &device);

    // 4. Run Rust Inference
    let config = &loaded.config.talker_config;
    let mut cache = (0..config.num_hidden_layers)
        .map(|_| KeyValueCache::new(1, config.num_key_value_heads, 512, config.head_dim))
        .collect::<Vec<_>>();

    let output = forward_talker_prefill(
        config,
        &loaded,
        TalkerForwardInput {
            inputs_embeds,
            position_ids,
            attention_mask: None,
            collect_activations: true,
        },
        &mut cache,
    )
    .expect("Rust inference failed");

    // 5. Assert Logits Alignment
    let actual_logits = output.logits;
    let layer_0 = output
        .activations
        .get("layers.0.hidden.output")
        .expect("layer 0 activation should be collected");
    let final_norm = output
        .activations
        .get("model.norm.output")
        .expect("final norm activation should be collected");
    compare_stats(
        "Layer0",
        layer_0.clone(),
        &ref_data.expected.layer_0_output,
        1.0,
        0.03125,
    );
    for layer_idx in 0..config.num_hidden_layers {
        let name = format!("layers.{layer_idx}.hidden.output");
        if let (Some(actual), Some(expected)) = (
            output.activations.get(&name),
            ref_data.expected.activations.get(&name),
        ) {
            compare_stats(&name, actual.clone(), expected, 8.0, 0.0625);
        }
    }
    compare_stats(
        "FinalNorm",
        final_norm.clone(),
        &ref_data.expected.final_norm,
        16.0,
        0.25,
    );
    assert_eq!(
        actual_logits.dims(),
        ref_data.expected.logits.shape.as_slice()
    );
    let actual_sum: f32 = actual_logits
        .clone()
        .cast(DType::F32)
        .sum()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()[0];

    println!("Python Logits Sum: {}", ref_data.expected.logits.sum);
    println!("Rust   Logits Sum: {}", actual_sum);

    let actual_first_5: Vec<f32> = actual_logits
        .clone()
        .flatten::<1>(0, 2)
        .slice([0..5])
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    println!("Python First 5: {:?}", ref_data.expected.logits.first_5);
    println!("Rust   First 5: {:?}", actual_first_5);

    if let Some(expected_values) = &ref_data.expected.logits.values {
        let actual_values: Vec<f32> = actual_logits
            .clone()
            .flatten::<1>(0, 2)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        assert_eq!(actual_values.len(), expected_values.len());
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
            let diff = (actual - expected).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = sum_abs_diff / actual_values.len() as f32;
        println!("Logits max abs diff: {max_abs_diff}");
        println!("Logits mean abs diff: {mean_abs_diff}");
        assert!(
            max_abs_diff < 0.75,
            "Logits max abs diff too large: {max_abs_diff}"
        );
        assert!(
            mean_abs_diff < 0.08,
            "Logits mean abs diff too large: {mean_abs_diff}"
        );
    }

    let diff = (actual_sum - ref_data.expected.logits.sum).abs();
    assert!(diff < 64.0, "Logits sum deviation too large: {}", diff);

    for (i, (a, e)) in actual_first_5
        .iter()
        .zip(ref_data.expected.logits.first_5.iter())
        .enumerate()
    {
        assert!(
            (a - e).abs() < 0.25,
            "Value mismatch at index {}: rust={}, py={}",
            i,
            a,
            e
        );
    }

    println!("Numerical alignment check PASSED!");
}

fn compare_stats(
    name: &str,
    tensor: Tensor<Backend, 3>,
    expected: &TensorStats,
    sum_tolerance: f32,
    first_values_tolerance: f32,
) {
    assert_eq!(
        tensor.dims(),
        expected.shape.as_slice(),
        "{name} shape mismatch"
    );
    let actual_sum: f32 = tensor
        .clone()
        .cast(DType::F32)
        .sum()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()[0];
    let actual_first_5 = tensor
        .flatten::<1>(0, 2)
        .slice([0..5])
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    println!("{name} Python Sum: {}", expected.sum);
    println!("{name} Rust   Sum: {}", actual_sum);
    println!("{name} Python First 5: {:?}", expected.first_5);
    println!("{name} Rust   First 5: {:?}", actual_first_5);
    let sum_diff = (actual_sum - expected.sum).abs();
    assert!(
        sum_diff < sum_tolerance,
        "{name} sum deviation too large: {sum_diff}"
    );
    for (idx, (actual, expected)) in actual_first_5
        .iter()
        .zip(expected.first_5.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        assert!(
            diff < first_values_tolerance,
            "{name} first_5[{idx}] mismatch: rust={actual}, py={expected}, diff={diff}"
        );
    }
}
