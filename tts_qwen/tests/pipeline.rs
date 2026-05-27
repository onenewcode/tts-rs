#![cfg(feature = "flex")]

mod common;

use burn::backend::Flex;
use tts_qwen::{
    CustomVoiceRequest, EngineConfig, QwenTtsEngine, SessionConfig, StepOutcome, StreamingMode,
};

type Backend = Flex;

#[test]
fn engine_loads_and_runs_a_session_to_completion() {
    let model_dir = common::resolve_model_dir();
    let device = Default::default();
    let mut engine =
        QwenTtsEngine::<Backend>::load(&model_dir, &device, EngineConfig::default()).unwrap();
    let handle = engine
        .start_session(
            CustomVoiceRequest::new("你好，欢迎使用语音合成。"),
            SessionConfig {
                max_new_tokens: 4,
                streaming: StreamingMode::AudioChunks,
                ..SessionConfig::default()
            },
        )
        .unwrap();

    loop {
        if matches!(engine.step(handle).unwrap(), StepOutcome::Finished) {
            break;
        }
        let _ = engine.drain_events(handle).unwrap();
    }

    let finished = engine.finish_session(handle).unwrap();
    assert!(finished.sample_rate > 0);
    assert!(finished.generated_audio_steps > 0);
    assert!(finished.talker_token_count > 0);
    assert!(!finished.waveform_pcm.is_empty());
}

#[test]
#[ignore = "loads real Qwen weights and writes a temp-dir wav artifact"]
fn engine_generates_valid_wav_with_real_model() {
    let model_dir = common::resolve_model_dir();
    let output_dir = common::unique_output_dir("pipeline-e2e");
    std::fs::create_dir_all(&output_dir).expect("e2e output dir should exist");
    let wav_path = output_dir.join("0000.wav");

    let device = Default::default();
    let mut engine =
        QwenTtsEngine::<Backend>::load(&model_dir, &device, EngineConfig::default()).unwrap();
    let handle = engine
        .start_session(
            CustomVoiceRequest {
                text: "你好，欢迎使用语音合成。".to_string(),
                language: Some("Chinese".to_string()),
                speaker: Some("Vivian".to_string()),
            },
            SessionConfig {
                max_new_tokens: 64,
                ..SessionConfig::default()
            },
        )
        .unwrap();
    let output = engine.run_to_end(handle).unwrap();

    let waveform = output
        .waveform_pcm
        .iter()
        .map(|sample| f32::from(*sample) / 32767.0)
        .collect::<Vec<_>>();
    let tensor = burn::tensor::Tensor::<Backend, 3>::from_floats(waveform.as_slice(), &device)
        .reshape([1, 1, waveform.len()]);
    tts_qwen::save_wav(&tensor, &wav_path, output.sample_rate).unwrap();

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
