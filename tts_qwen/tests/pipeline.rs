mod common;

use burn::backend::Flex;
use tts_qwen::{CustomVoiceRequest, Qwen3TtsInferOptions, Qwen3TtsPipeline};

type Backend = Flex;

#[test]
fn pipeline_load_reports_are_available_through_the_facade() {
    let model_dir = common::resolve_model_dir();
    let device = Default::default();
    let pipeline = Qwen3TtsPipeline::<Backend>::load(&model_dir, &device).unwrap();
    let load_report = pipeline.load_report();

    assert!(load_report.talker.applied > 0);
    assert!(load_report.audio_codec.applied > 0);
    assert_eq!(
        pipeline.generation_config().codec_eos_token_id,
        2150,
        "facade should surface the resolved codec EOS token"
    );
    assert!(
        pipeline
            .generation_config()
            .suppress_token_ids
            .contains(&2148),
        "facade should surface reserved suppress tokens"
    );
}

#[test]
#[ignore = "loads real Qwen weights and writes target/tmp/e2e/0000.wav"]
fn pipeline_generates_valid_wav_with_real_model() {
    let model_dir = common::resolve_model_dir();
    let output_dir = common::workspace_root().join("target/tmp/e2e");
    std::fs::create_dir_all(&output_dir).expect("e2e output dir should exist");
    let wav_path = output_dir.join("0000.wav");

    let device = Default::default();
    let pipeline = Qwen3TtsPipeline::<Backend>::load(&model_dir, &device).unwrap();
    let request = CustomVoiceRequest {
        text: "你好，欢迎使用语音合成。".to_string(),
        language: Some("Chinese".to_string()),
        speaker: Some("Vivian".to_string()),
    };
    let output = pipeline
        .infer_to_wav(
            &request,
            &Qwen3TtsInferOptions {
                max_new_tokens: 64,
                ..Qwen3TtsInferOptions::default()
            },
            &wav_path,
        )
        .expect("pipeline should generate wav");

    assert!(
        output.codec_generation.generated_audio_steps > 0,
        "talker should emit at least one audio token"
    );
    let generated_ids = output
        .codec_generation
        .talker_token_ids
        .clone()
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    assert!(
        generated_ids.len() <= 64,
        "talker generation should honor max_new_tokens"
    );

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
    assert!(
        bytes[44..].chunks_exact(2).any(|sample| sample != [0, 0]),
        "audio should not be all zero"
    );
}
