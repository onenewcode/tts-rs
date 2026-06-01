use std::path::PathBuf;

use tts_app::{
    BaseSynthesisInput, CustomVoiceSynthesisInput, QwenAppService, SharedSynthesisInput,
};
use tts_qwen3_tts::{
    BaseVoiceCloneConditioning, LanguageSelection, Qwen3TtsPackageSource, QwenRequest,
};

#[test]
fn base_request_requires_ref_text_when_reference_audio_is_not_x_vector_only() {
    let error = QwenAppService::build_base_request(BaseSynthesisInput {
        shared: SharedSynthesisInput {
            model_dir: Some(PathBuf::from("model-dir")),
            manifest: None,
            text: "hello".to_string(),
            language: "auto".to_string(),
            output: PathBuf::from("out.wav"),
            max_new_tokens: None,
            sampling: tts_qwen3_tts::SamplingConfig::greedy(),
            profiling: false,
            profiling_per_step: false,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 8,
        },
        ref_audio: Some(PathBuf::from("clone.wav")),
        ref_text: None,
        x_vector_only: false,
    })
    .unwrap_err();

    assert!(error.to_string().contains("ref-text"));
}

#[test]
fn base_request_building_moves_shell_semantics_out_of_cli() {
    let prepared = QwenAppService::prepare_base(BaseSynthesisInput {
        shared: SharedSynthesisInput {
            model_dir: None,
            manifest: Some(PathBuf::from("package.yaml")),
            text: "hello".to_string(),
            language: "zh".to_string(),
            output: PathBuf::from("out.wav"),
            max_new_tokens: None,
            sampling: tts_qwen3_tts::SamplingConfig::greedy(),
            profiling: false,
            profiling_per_step: false,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 8,
        },
        ref_audio: Some(PathBuf::from("clone.wav")),
        ref_text: Some("reference words".to_string()),
        x_vector_only: false,
    })
    .unwrap();

    assert!(matches!(
        prepared.package_source,
        Qwen3TtsPackageSource::ManifestPath(_)
    ));
    match prepared.request {
        QwenRequest::Base(request) => {
            assert_eq!(request.language, LanguageSelection::Named("zh".to_string()));
            assert!(matches!(
                request.voice_clone,
                Some(BaseVoiceCloneConditioning::ReferenceAudio(_))
            ));
        }
        QwenRequest::CustomVoice(_) => panic!("expected base request"),
    }
}

#[test]
fn custom_voice_request_building_preserves_driver_specific_fields() {
    let prepared = QwenAppService::prepare_custom_voice(CustomVoiceSynthesisInput {
        shared: SharedSynthesisInput {
            model_dir: Some(PathBuf::from("model-dir")),
            manifest: None,
            text: "hello".to_string(),
            language: "auto".to_string(),
            output: PathBuf::from("out.wav"),
            max_new_tokens: None,
            sampling: tts_qwen3_tts::SamplingConfig::greedy(),
            profiling: false,
            profiling_per_step: false,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 8,
        },
        speaker: "Chelsie".to_string(),
        instruct: Some("cheerful".to_string()),
    })
    .unwrap();

    assert!(matches!(
        prepared.package_source,
        Qwen3TtsPackageSource::ModelDir(_)
    ));
    match prepared.request {
        QwenRequest::CustomVoice(request) => {
            assert_eq!(request.language, LanguageSelection::Auto);
            assert_eq!(request.speaker.as_deref(), Some("Chelsie"));
            assert_eq!(request.instruct.as_deref(), Some("cheerful"));
        }
        QwenRequest::Base(_) => panic!("expected custom voice request"),
    }
}
