use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use tts_rs_qwen_burn::{
    KeyValueCache, TalkerDecodeInput, TalkerForwardInput, forward_talker_decode_step,
    forward_talker_prefill, load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;

#[derive(Deserialize)]
struct ReferenceData {
    input: InputData,
    decode_input: Option<DecodeInputData>,
    expected: ExpectedData,
    decode_expected: Option<DecodeExpectedData>,
}

#[derive(Deserialize)]
struct InputData {
    inputs_embeds: Vec<Vec<Vec<f32>>>,
    position_ids: Vec<Vec<Vec<i32>>>,
}

#[derive(Deserialize)]
struct DecodeInputData {
    inputs_embeds: Vec<Vec<Vec<f32>>>,
    position_ids: Vec<Vec<Vec<i32>>>,
    attention_mask: Vec<Vec<i32>>,
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
struct DecodeExpectedData {
    logits: TensorStats,
    last_hidden_state: TensorStats,
    cache_len_before: usize,
    cache_len_after: usize,
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
    let input_dims = dims3_f32(&ref_data.input.inputs_embeds);
    let inputs_embeds =
        Tensor::<Backend, 3>::from_data(TensorData::new(flattened_embeds, input_dims), &device)
            .cast(DType::BF16); // Match model weights

    let flattened_pos: Vec<i32> = ref_data
        .input
        .position_ids
        .iter()
        .flatten()
        .flatten()
        .cloned()
        .collect();
    let position_ids = Tensor::<Backend, 3, Int>::from_data(
        TensorData::new(flattened_pos, dims3_i32(&ref_data.input.position_ids)),
        &device,
    );

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
        compare_full_values("Logits", actual_logits.clone(), expected_values);
    }

    let diff = (actual_sum - ref_data.expected.logits.sum).abs();
    assert!(diff < 512.0, "Logits sum deviation too large: {}", diff);

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

    if let (Some(decode_input), Some(decode_expected)) =
        (&ref_data.decode_input, &ref_data.decode_expected)
    {
        assert_eq!(
            cache[0].len(),
            decode_expected.cache_len_before,
            "decode cache length before step"
        );

        let decode_embeds = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                decode_input
                    .inputs_embeds
                    .iter()
                    .flatten()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims3_f32(&decode_input.inputs_embeds),
            ),
            &device,
        )
        .cast(DType::BF16);
        let decode_position_ids = Tensor::<Backend, 3, Int>::from_data(
            TensorData::new(
                decode_input
                    .position_ids
                    .iter()
                    .flatten()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims3_i32(&decode_input.position_ids),
            ),
            &device,
        );
        let decode_attention_mask = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(
                decode_input
                    .attention_mask
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims2_i32(&decode_input.attention_mask),
            ),
            &device,
        );

        let decode_output = forward_talker_decode_step(
            config,
            &loaded,
            TalkerDecodeInput {
                inputs_embeds: decode_embeds,
                position_ids: decode_position_ids,
                attention_mask: Some(decode_attention_mask),
                collect_activations: true,
            },
            &mut cache,
        )
        .expect("Rust decode inference failed");

        assert_eq!(
            cache[0].len(),
            decode_expected.cache_len_after,
            "decode cache length after step"
        );
        for layer_idx in 0..config.num_hidden_layers {
            let name = format!("layers.{layer_idx}.hidden.output");
            if let (Some(actual), Some(expected)) = (
                decode_output.activations.get(&name),
                decode_expected.activations.get(&name),
            ) {
                compare_stats(
                    &format!("Decode {name}"),
                    actual.clone(),
                    expected,
                    8.0,
                    0.125,
                );
            }
        }
        compare_stats(
            "DecodeFinalNorm",
            decode_output.last_hidden_state.clone(),
            &decode_expected.last_hidden_state,
            16.0,
            0.75,
        );
        compare_stats(
            "DecodeLogits",
            decode_output.logits.clone(),
            &decode_expected.logits,
            512.0,
            0.75,
        );

        if let Some(expected_values) = &decode_expected.logits.values {
            compare_full_values_with_tolerance(
                "Decode logits",
                decode_output.logits.clone(),
                expected_values,
                2.0,
                0.2,
            );
        }
    } else {
        println!("Decode alignment skipped: reference.json has no V2 decode case");
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
            diff <= first_values_tolerance,
            "{name} first_5[{idx}] mismatch: rust={actual}, py={expected}, diff={diff}"
        );
    }
}

fn compare_full_values(name: &str, tensor: Tensor<Backend, 3>, expected_values: &[f32]) {
    compare_full_values_with_tolerance(name, tensor, expected_values, 0.75, 0.08);
}

fn compare_full_values_with_tolerance(
    name: &str,
    tensor: Tensor<Backend, 3>,
    expected_values: &[f32],
    max_abs_tolerance: f32,
    mean_abs_tolerance: f32,
) {
    let actual_values: Vec<f32> = tensor
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
    println!("{name} max abs diff: {max_abs_diff}");
    println!("{name} mean abs diff: {mean_abs_diff}");
    assert!(
        max_abs_diff < max_abs_tolerance,
        "{name} max abs diff too large: {max_abs_diff}"
    );
    assert!(
        mean_abs_diff < mean_abs_tolerance,
        "{name} mean abs diff too large: {mean_abs_diff}"
    );
}

fn dims3_f32(values: &[Vec<Vec<f32>>]) -> [usize; 3] {
    [
        values.len(),
        values.first().map_or(0, Vec::len),
        values
            .first()
            .and_then(|batch| batch.first())
            .map_or(0, Vec::len),
    ]
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
