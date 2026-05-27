mod common;

use burn::backend::Flex;
use tts_qwen::{CustomVoiceRequest, LocalInferenceCore, LocalInferenceOptions, QwenTtsAdapter};

type Backend = Flex;

#[test]
fn core_load_reports_are_available_through_qwen_adapter() {
    let model_dir = common::resolve_model_dir();
    let device = Default::default();
    let core = LocalInferenceCore::<Backend, QwenTtsAdapter<Backend>>::load(&model_dir, &device)
        .expect("core should load");
    let load_report = core.load_report();

    assert!(load_report.talker.applied > 0);
    assert!(load_report.audio_codec.applied > 0);
}

#[test]
#[ignore = "loads real Qwen weights and validates the generic core profile path"]
fn core_infer_to_file_collects_profile_stages() {
    let model_dir = common::resolve_model_dir();
    let output_dir = common::workspace_root().join("target/tmp/core-e2e");
    std::fs::create_dir_all(&output_dir).expect("core e2e output dir should exist");
    let wav_path = output_dir.join("0000.wav");

    let device = Default::default();
    let core = LocalInferenceCore::<Backend, QwenTtsAdapter<Backend>>::load(&model_dir, &device)
        .expect("core should load");
    let request = CustomVoiceRequest {
        text: "你好，欢迎使用语音合成。".to_string(),
        language: Some("Chinese".to_string()),
        speaker: Some("Vivian".to_string()),
    };
    let run = core
        .infer_to_file(
            &request,
            &LocalInferenceOptions {
                max_new_tokens: 64,
                ..LocalInferenceOptions::default()
            },
            &wav_path,
        )
        .expect("core should generate wav");

    assert!(run.output.codec_generation.generated_audio_steps > 0);
    assert!(run.profile.total_elapsed_ms > 0);
    assert!(
        run.profile
            .stages
            .iter()
            .any(|stage| stage.name == "frontend_build")
    );
    assert!(
        run.profile
            .stages
            .iter()
            .any(|stage| stage.name == "talker_generation")
    );
    assert!(
        run.profile
            .stages
            .iter()
            .any(|stage| stage.name == "audio_decode")
    );
    assert!(
        run.profile
            .stages
            .iter()
            .any(|stage| stage.name == "wav_write")
    );
}
