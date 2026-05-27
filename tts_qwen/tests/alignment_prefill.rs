mod common;

use std::process::Command;

use burn::backend::Flex;
use serde::Deserialize;
use tts_qwen::{
    CustomVoiceBatch, CustomVoiceRequest, Qwen3TtsTextTokenizer, build_custom_voice_prefill_batch,
    load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;

#[derive(Debug, Deserialize)]
struct TensorReference {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct PrefillReference {
    text: String,
    language: String,
    speaker: String,
    text_token_ids: Vec<i64>,
    codec_prefix_ids: Vec<i64>,
    attention_mask: Vec<Vec<i32>>,
    position_ids: Vec<Vec<Vec<i32>>>,
    inputs_embeds: TensorReference,
    trailing_text_hidden: TensorReference,
    tts_pad_embed: TensorReference,
}

#[test]
fn prefill_matches_python_oracle() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_prefill.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_prefill.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python prefill oracle");
    assert!(status.success(), "Python prefill oracle failed");

    let reference: PrefillReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();
    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device)
        .expect("talker should load");
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();
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
    .expect("frontend prefill should build");

    assert_eq!(frontend.text_token_ids, vec![reference.text_token_ids]);
    assert_eq!(frontend.codec_prefix_ids, vec![reference.codec_prefix_ids]);
    assert_eq!(
        tensor_i32_2d(frontend.attention_mask),
        reference
            .attention_mask
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        tensor_i32_3d(frontend.position_ids),
        reference
            .position_ids
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<_>>()
    );
    assert_tensor_close(
        "inputs_embeds",
        frontend.inputs_embeds,
        reference.inputs_embeds,
    );
    assert_tensor_close(
        "trailing_text_hidden",
        frontend.trailing_text_hidden,
        reference.trailing_text_hidden,
    );
    assert_tensor_close(
        "tts_pad_embed",
        frontend.tts_pad_embed,
        reference.tts_pad_embed,
    );
}

fn tensor_i32_2d(tensor: burn::tensor::Tensor<Backend, 2, burn::tensor::Int>) -> Vec<i32> {
    tensor
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap()
}

fn tensor_i32_3d(tensor: burn::tensor::Tensor<Backend, 3, burn::tensor::Int>) -> Vec<i32> {
    tensor
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap()
}

fn assert_tensor_close(
    name: &str,
    tensor: burn::tensor::Tensor<Backend, 3>,
    reference: TensorReference,
) {
    assert_eq!(
        tensor.dims().as_slice(),
        reference.shape.as_slice(),
        "{name} shape"
    );
    let actual = tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(actual.len(), reference.values.len(), "{name} len");
    let max_abs = actual
        .iter()
        .zip(reference.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= 5e-2, "{name} max_abs={max_abs}");
}
