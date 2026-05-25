use burn_wgpu::Wgpu;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use tts_rs_qwen_burn::{
    CodePredictorGenerateInput, KeyValueCache, TalkerDecodeInput, TalkerForwardInput,
    TalkerGenerateInput, forward_talker_decode_step, forward_talker_prefill,
    generate_code_predictor_groups, generate_talker_tokens,
    load_qwen3_tts_talker_for_inference,
};

mod common;

type Backend = Wgpu;
const ABS_TOLERANCE: f32 = 0.005;

#[derive(Deserialize)]
struct ReferenceData {
    input: InputData,
    decode_input: Option<DecodeInputData>,
    expected: ExpectedData,
    decode_expected: Option<DecodeExpectedData>,
    generation_input: Option<GenerationInputData>,
    generation_expected: Option<GenerationExpectedData>,
    code_predictor_generation_input: Option<CodePredictorGenerationInputData>,
    code_predictor_generation_expected: Option<CodePredictorGenerationExpectedData>,
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
struct GenerationInputData {
    max_new_tokens: usize,
}

#[derive(Deserialize)]
struct GenerationExpectedData {
    generated_token_ids: Vec<Vec<i32>>,
    prefill_selected_token_id: Vec<i32>,
    steps: Vec<GenerationStepExpectedData>,
}

#[derive(Deserialize)]
struct GenerationStepExpectedData {
    token_id: Vec<i32>,
    logits: TensorStats,
    cache_len_before: usize,
    cache_len_after: usize,
}

#[derive(Deserialize)]
struct CodePredictorGenerationInputData {
    talker_hidden_state: Vec<Vec<f32>>,
    base_codec_token_id: Vec<Vec<i32>>,
}

#[derive(Deserialize)]
struct CodePredictorGenerationExpectedData {
    codec_ids: Vec<Vec<i32>>,
    predictor_token_ids: Vec<Vec<i32>>,
    steps: Vec<CodePredictorGenerationStepExpectedData>,
}

#[derive(Deserialize)]
struct CodePredictorGenerationStepExpectedData {
    token_id: Vec<i32>,
    logits: TensorStats,
    cache_len_before: usize,
    cache_len_after: usize,
}

#[derive(Deserialize)]
struct TensorStats {
    shape: Vec<usize>,
    first_5: Vec<f32>,
    last_5: Vec<f32>,
    values: Option<Vec<f32>>,
}

#[test]
#[ignore] // Run manually with `cargo test --test talker_alignment -- --ignored`
fn test_numerical_alignment_with_python_reference() {
    let device = Default::default();

    // 1. Load reference data
    let ref_path = common::workspace_root().join("reference.json");
    let ref_json = fs::read_to_string(ref_path)
        .expect("reference.json not found. Run `python py/generate_reference.py` first.");
    let ref_data: ReferenceData = serde_json::from_str(&ref_json).unwrap();

    // 2. Load Rust Model
    let model_dir = common::resolve_model_dir();
    let loaded = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device)
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
    for layer_idx in 0..config.num_hidden_layers {
        for suffix in [
            "input_norm.output",
            "attn.output",
            "attn_residual.output",
            "post_attention_norm.output",
            "mlp.gate",
            "mlp.up",
            "mlp.activated_gate",
            "mlp.product",
            "mlp.output",
            "hidden.output",
        ] {
            let name = format!("layers.{layer_idx}.{suffix}");
            if let (Some(actual), Some(expected)) = (
                output.activations.get(&name),
                ref_data.expected.activations.get(&name),
            ) {
                compare_tensor_values(&name, actual.clone(), expected);
            }
        }
    }
    compare_tensor_values("Layer0", layer_0.clone(), &ref_data.expected.layer_0_output);
    compare_tensor_values(
        "FinalNorm",
        final_norm.clone(),
        &ref_data.expected.final_norm,
    );
    compare_tensor_values("Logits", actual_logits.clone(), &ref_data.expected.logits);

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
                compare_tensor_values(&format!("Decode {name}"), actual.clone(), expected);
            }
        }
        compare_tensor_values(
            "DecodeFinalNorm",
            decode_output.last_hidden_state.clone(),
            &decode_expected.last_hidden_state,
        );
        compare_tensor_values(
            "DecodeLogits",
            decode_output.logits.clone(),
            &decode_expected.logits,
        );
    } else {
        println!("Decode alignment skipped: reference.json has no V2 decode case");
    }

    if let (Some(generation_input), Some(generation_expected)) =
        (&ref_data.generation_input, &ref_data.generation_expected)
    {
        let generation_embeds = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                ref_data
                    .input
                    .inputs_embeds
                    .iter()
                    .flatten()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims3_f32(&ref_data.input.inputs_embeds),
            ),
            &device,
        )
        .cast(DType::BF16);
        let generation_position_ids = Tensor::<Backend, 3, Int>::from_data(
            TensorData::new(
                ref_data
                    .input
                    .position_ids
                    .iter()
                    .flatten()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims3_i32(&ref_data.input.position_ids),
            ),
            &device,
        );
        let mut generation_cache = (0..config.num_hidden_layers)
            .map(|_| KeyValueCache::new(1, config.num_key_value_heads, 512, config.head_dim))
            .collect::<Vec<_>>();

        let generation_output = generate_talker_tokens(
            config,
            &loaded,
            TalkerGenerateInput {
                prefill_inputs_embeds: generation_embeds,
                prefill_position_ids: generation_position_ids,
                prefill_attention_mask: None,
                max_new_tokens: generation_input.max_new_tokens,
                collect_step_diagnostics: true,
            },
            &mut generation_cache,
        )
        .expect("Rust generation failed");

        let actual_token_ids = generation_output
            .generated_token_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        let expected_token_ids = generation_expected
            .generated_token_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            actual_token_ids, expected_token_ids,
            "generated token ids should match Python greedy generation"
        );

        let actual_prefill_token = generation_output
            .generated_token_ids
            .clone()
            .slice([0..1, 0..1])
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        assert_eq!(
            actual_prefill_token, generation_expected.prefill_selected_token_id,
            "prefill selected token should come from the last prefill logits"
        );

        assert_eq!(
            generation_output.step_logits.len(),
            generation_expected.steps.len(),
            "generation decode step count"
        );
        for (idx, (actual_logits, expected_step)) in generation_output
            .step_logits
            .iter()
            .zip(generation_expected.steps.iter())
            .enumerate()
        {
            let actual_step_token = generation_output
                .generated_token_ids
                .clone()
                .slice([0..1, idx + 1..idx + 2])
                .into_data()
                .convert::<i32>()
                .into_vec::<i32>()
                .unwrap();
            assert_eq!(
                actual_step_token, expected_step.token_id,
                "generation step {idx} selected token"
            );
            assert_eq!(
                generation_output.step_diagnostics[idx].cache_len_before,
                expected_step.cache_len_before,
                "generation step {idx} cache length before"
            );
            assert_eq!(
                generation_output.step_diagnostics[idx].cache_len_after,
                expected_step.cache_len_after,
                "generation step {idx} cache length after"
            );
            compare_tensor_values(
                &format!("Generation step {idx} logits"),
                actual_logits.clone(),
                &expected_step.logits,
            );
        }
    } else {
        println!("Generation alignment skipped: reference.json has no V3 generation case");
    }

    if let (Some(code_predictor_input), Some(code_predictor_expected)) = (
        &ref_data.code_predictor_generation_input,
        &ref_data.code_predictor_generation_expected,
    ) {
        let talker_hidden_state = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                code_predictor_input
                    .talker_hidden_state
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims2_f32(&code_predictor_input.talker_hidden_state),
            ),
            &device,
        )
        .cast(DType::BF16);
        let base_codec_token_id = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(
                code_predictor_input
                    .base_codec_token_id
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                dims2_i32(&code_predictor_input.base_codec_token_id),
            ),
            &device,
        );
        let predictor_config = &config.code_predictor_config;
        let mut predictor_cache = (0..predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    predictor_config.num_key_value_heads,
                    512,
                    predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();

        let predictor_output = generate_code_predictor_groups(
            config,
            &loaded,
            CodePredictorGenerateInput {
                talker_hidden_state,
                base_codec_token_id,
                collect_step_diagnostics: true,
            },
            &mut predictor_cache,
        )
        .expect("Rust code predictor generation failed");

        let actual_codec_ids = predictor_output
            .codec_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        let expected_codec_ids = code_predictor_expected
            .codec_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            actual_codec_ids, expected_codec_ids,
            "code predictor generated codec ids should match Python"
        );

        let actual_predictor_token_ids = predictor_output
            .predictor_token_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        let expected_predictor_token_ids = code_predictor_expected
            .predictor_token_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            actual_predictor_token_ids, expected_predictor_token_ids,
            "code predictor generated predictor ids should match Python"
        );

        assert_eq!(
            predictor_output.step_logits.len(),
            code_predictor_expected.steps.len(),
            "code predictor generation step count"
        );
        for (idx, (actual_logits, expected_step)) in predictor_output
            .step_logits
            .iter()
            .zip(code_predictor_expected.steps.iter())
            .enumerate()
        {
            let actual_step_token = predictor_output
                .predictor_token_ids
                .clone()
                .slice([0..1, idx..idx + 1])
                .into_data()
                .convert::<i32>()
                .into_vec::<i32>()
                .unwrap();
            assert_eq!(
                actual_step_token, expected_step.token_id,
                "code predictor generation step {idx} selected token"
            );
            assert_eq!(
                predictor_output.step_diagnostics[idx].cache_len_before,
                expected_step.cache_len_before,
                "code predictor generation step {idx} cache length before"
            );
            assert_eq!(
                predictor_output.step_diagnostics[idx].cache_len_after,
                expected_step.cache_len_after,
                "code predictor generation step {idx} cache length after"
            );
            compare_tensor_values(
                &format!("Code predictor generation step {idx} logits"),
                actual_logits.clone(),
                &expected_step.logits,
            );
        }
    } else {
        println!("Code predictor generation alignment skipped: reference.json has no V4 case");
    }

    println!("Numerical alignment check PASSED!");
}

fn compare_tensor_values(name: &str, tensor: Tensor<Backend, 3>, expected: &TensorStats) {
    assert_eq!(
        tensor.dims(),
        expected.shape.as_slice(),
        "{name} shape mismatch"
    );
    let actual_values = tensor
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    let actual_first_5 = actual_values.iter().take(5).copied().collect::<Vec<_>>();
    let mut actual_last_5 = actual_values
        .iter()
        .rev()
        .take(5)
        .copied()
        .collect::<Vec<_>>();
    actual_last_5.reverse();

    println!("{name} Python First 5: {:?}", expected.first_5);
    println!("{name} Rust   First 5: {:?}", actual_first_5);
    println!("{name} Python Last 5: {:?}", expected.last_5);
    println!("{name} Rust   Last 5: {:?}", actual_last_5);

    let expected_values = expected
        .values
        .as_ref()
        .expect("reference TensorStats must include full flattened values");
    compare_full_tensor(name, &actual_values, expected_values);
}

fn compare_full_tensor(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name} flattened tensor length mismatch"
    );
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_abs_idx = idx;
        }
        assert!(
            diff <= ABS_TOLERANCE,
            "{name}[{idx}] mismatch: rust={actual}, py={expected}, diff={diff}, tolerance={ABS_TOLERANCE}, max_abs_so_far={max_abs_diff} at {max_abs_idx}"
        );
    }
    println!("{name} full tensor max_abs_diff={max_abs_diff} at {max_abs_idx}");
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

fn dims2_f32(values: &[Vec<f32>]) -> [usize; 2] {
    [values.len(), values.first().map_or(0, Vec::len)]
}

fn dims2_i32(values: &[Vec<i32>]) -> [usize; 2] {
    [values.len(), values.first().map_or(0, Vec::len)]
}
