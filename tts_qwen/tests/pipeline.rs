#![cfg(feature = "flex")]

mod common;

use tts_core::{ComputeBackend, ModelRegistry, SynthesisOptions, SynthesisRequest, TtsService};
use tts_qwen::register_qwen_family_model;

#[test]
fn engine_loads_and_runs_a_session_to_completion() {
    let model_dir = common::resolve_model_dir();
    let mut registry = ModelRegistry::new();
    assert!(register_qwen_family_model(
        &mut registry,
        "qwen-test",
        &model_dir,
        "qwen3-tts-12hz-0.6b-customvoice",
    ));
    let service = TtsService::new(registry);
    let finished = service
        .synthesize(
            "qwen-test",
            &SynthesisRequest {
                text: "你好，欢迎使用语音合成。".to_string(),
                language: None,
                speaker: None,
            },
            &SynthesisOptions {
                max_new_tokens: 4,
                stream: true,
                backend: Some(ComputeBackend::Flex),
                ..SynthesisOptions::default()
            },
        )
        .unwrap();
    assert!(finished.sample_rate > 0);
    assert!(!finished.waveform_pcm.is_empty());
}

#[test]
#[ignore = "loads real Qwen weights and writes a temp-dir wav artifact"]
fn engine_generates_valid_wav_with_real_model() {
    let model_dir = common::resolve_model_dir();
    let output_dir = common::unique_output_dir("pipeline-e2e");
    std::fs::create_dir_all(&output_dir).expect("e2e output dir should exist");
    let wav_path = output_dir.join("0000.wav");
    let mut registry = ModelRegistry::new();
    assert!(register_qwen_family_model(
        &mut registry,
        "qwen-test",
        &model_dir,
        "qwen3-tts-12hz-0.6b-customvoice",
    ));
    let service = TtsService::new(registry);
    let output = service
        .synthesize(
            "qwen-test",
            &SynthesisRequest {
                text: "你好，欢迎使用语音合成。".to_string(),
                language: Some("Chinese".to_string()),
                speaker: Some("Vivian".to_string()),
            },
            &SynthesisOptions {
                max_new_tokens: 64,
                backend: Some(ComputeBackend::Flex),
                ..SynthesisOptions::default()
            },
        )
        .unwrap();

    tts_core::save_pcm_wav(&output.waveform_pcm, &wav_path, output.sample_rate).unwrap();

    let wav = std::fs::read(&wav_path).expect("wav should be readable");
    assert_valid_nonempty_wav(&wav, output.sample_rate);
}

fn assert_valid_nonempty_wav(bytes: &[u8], expected_sample_rate: u32) {
    assert!(bytes.len() > 44, "wav must include header and PCM data");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "PCM format");
    assert_eq!(
        u16::from_le_bytes([bytes[22], bytes[23]]),
        1,
        "mono channel"
    );
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        expected_sample_rate
    );
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "16-bit PCM");
    assert_eq!(&bytes[36..40], b"data");
    let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    assert_eq!(bytes.len(), 44 + data_size);
    assert!(data_size > 0);
}
