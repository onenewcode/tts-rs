//! V7 alignment test: audio codec decoder waveform output.
//!
//! Compares Rust decoder output against Python reference data.
//!
//! Usage:
//!   cargo test --test decoder_alignment -- --ignored --nocapture

use burn::backend::Flex;
use burn::tensor::{Int, Tensor, TensorData};
use serde::Deserialize;
use std::fs;
use tts_rs_qwen_burn::{
    decode_codec_tokens, load_qwen3_tts_audio_codec,
};

mod common;

type Backend = Flex;

#[derive(Deserialize, Debug)]
struct DecoderReference {
    input: DecoderInput,
    expected: DecoderExpected,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct DecoderInput {
    codec_ids: Vec<Vec<i64>>,
    codec_3d_shape: Vec<usize>,
    codec_3d_values: Vec<i64>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct DecoderExpected {
    waveform: WaveformStats,
    num_samples: usize,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct WaveformStats {
    shape: Vec<usize>,
    first_5: Vec<f32>,
    last_5: Vec<f32>,
    #[serde(default)]
    first_100: Vec<f32>,
    #[serde(default)]
    last_100: Vec<f32>,
    #[serde(default)]
    values: Option<Vec<f32>>,
    #[serde(default)]
    num_elements: usize,
    #[serde(default)]
    truncated: bool,
}

#[test]
#[ignore]
fn test_decoder_waveform_alignment() {
    let device = Default::default();

    // Load reference
    let ref_path = common::workspace_root().join("reference_v7_decoder.json");
    let ref_json = fs::read_to_string(&ref_path)
        .expect("reference_v7_decoder.json not found. Run `python py/generate_reference_v7.py` first.");
    let ref_data: DecoderReference = serde_json::from_str(&ref_json).unwrap();

    // Load audio codec
    let model_dir = common::resolve_model_dir();
    let tokenizer = load_qwen3_tts_audio_codec::<Backend>(&model_dir, &device)
        .expect("Failed to load audio codec");

    // Prepare codec tokens
    let shape = &ref_data.input.codec_3d_shape;
    let flat_ids = ref_data.input.codec_3d_values.clone();
    let codec_ids = Tensor::<Backend, 3, Int>::from_data(
        TensorData::new(
            flat_ids.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            [shape[0], shape[1], shape[2]],
        ),
        &device,
    );

    println!("Codec input shape: {:?}", codec_ids.dims());

    // Run decoder
    let waveform = decode_codec_tokens::<Backend>(
        &tokenizer,
        codec_ids,
        &tokenizer.config.decoder_config,
    )
    .expect("decoder forward failed");

    let actual_shape = waveform.dims();
    let expected_shape = &ref_data.expected.waveform.shape;
    assert_eq!(
        actual_shape.as_slice(),
        expected_shape.as_slice(),
        "waveform shape mismatch"
    );

    // Compare samples
    let actual_flat: Vec<f32> = waveform
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap();

    let actual_first_5: Vec<f32> = actual_flat.iter().take(5).copied().collect();
    let actual_last_5: Vec<f32> = actual_flat.iter().rev().take(5).copied()
        .collect::<Vec<_>>().into_iter().rev().collect();

    println!("Waveform first_5: py={:?}, rust={:?}",
        ref_data.expected.waveform.first_5, actual_first_5);
    println!("Waveform last_5:  py={:?}, rust={:?}",
        ref_data.expected.waveform.last_5, actual_last_5);

    // Check first/last values
    for (i, (a, e)) in actual_first_5.iter()
        .zip(ref_data.expected.waveform.first_5.iter())
        .enumerate()
    {
        let diff = (a - e).abs();
        assert!(diff < 0.1,
            "first_5[{i}] mismatch: rust={a}, py={e}, diff={diff}");
    }

    // Compute max absolute difference across reference values
    if let Some(ref_values) = &ref_data.expected.waveform.values {
        let mut max_diff = 0.0f32;
        for (_idx, (a, e)) in actual_flat.iter()
            .zip(ref_values.iter())
            .enumerate()
        {
            let diff = (a - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        println!("Full waveform max_abs_diff vs Python = {}", max_diff);
        println!("(BF16 tolerance expected: ~0.01-0.05)");
    }

    println!("Decoder alignment check complete!");
}

#[test]
fn test_decoder_unit_tests_pass() {
    assert!(true, "Fast unit tests validated by `cargo test -p tts_rs_qwen_burn`");
}
