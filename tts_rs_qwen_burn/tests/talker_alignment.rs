use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
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
}

#[derive(Deserialize)]
struct TensorStats {
    shape: Vec<usize>,
    sum: f32,
    first_5: Vec<f32>,
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
            collect_activations: false,
        },
        &mut cache,
    )
    .expect("Rust inference failed");

    // 5. Assert Logits Alignment
    let actual_logits = output.logits;
    let actual_sum: f32 = actual_logits
        .clone()
        .sum()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()[0];

    println!("Python Logits Sum: {}", ref_data.expected.logits.sum);
    println!("Rust   Logits Sum: {}", actual_sum);

    let diff = (actual_sum - ref_data.expected.logits.sum).abs();
    // Allow for small deviation
    assert!(diff < 2.0, "Logits sum deviation too large: {}", diff);

    let actual_first_5: Vec<f32> = actual_logits
        .flatten::<1>(0, 2)
        .slice([0..5])
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    for (i, (a, e)) in actual_first_5
        .iter()
        .zip(ref_data.expected.logits.first_5.iter())
        .enumerate()
    {
        assert!(
            (a - e).abs() < 1.0,
            "Value mismatch at index {}: rust={}, py={}",
            i,
            a,
            e
        );
    }

    println!("Numerical alignment check PASSED!");
}
