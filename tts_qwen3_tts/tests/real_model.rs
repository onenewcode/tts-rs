#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use tts_qwen3_tts::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    LanguageSelection, Qwen3TtsBackend, Qwen3TtsEngine, Qwen3TtsEngineConfig,
    Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

#[test]
#[ignore = "loads the in-crate runtime with real custom-voice model assets"]
fn custom_voice_speaker_smoke_generates_pcm_audio() {
    let model_dir = custom_voice_model_dir();
    if !model_dir.join("config.json").is_file() {
        return;
    }

    let engine = load_engine(&model_dir);
    let audio = engine
        .synthesize(
            QwenRequest::CustomVoice(CustomVoiceRequest {
                text: "你好，欢迎使用语音合成。".to_string(),
                language: LanguageSelection::Named("Chinese".to_string()),
                speaker: Some("Vivian".to_string()),
                instruct: None,
            }),
            smoke_options(),
        )
        .expect("custom-voice speaker path should synthesize audio");

    assert_pcm_audio(&audio);
}

#[test]
#[ignore = "loads the in-crate runtime with real custom-voice model assets"]
fn custom_voice_instruct_smoke_generates_pcm_audio() {
    let model_dir = custom_voice_model_dir();
    if !model_dir.join("config.json").is_file() {
        return;
    }

    let engine = load_engine(&model_dir);
    let audio = engine
        .synthesize(
            QwenRequest::CustomVoice(CustomVoiceRequest {
                text: "你好，欢迎使用语音合成。".to_string(),
                language: LanguageSelection::Named("Chinese".to_string()),
                speaker: Some("Vivian".to_string()),
                instruct: Some("用特别愤怒的语气说".to_string()),
            }),
            smoke_options(),
        )
        .expect("custom-voice instruct path should synthesize audio");

    assert_pcm_audio(&audio);
}

#[test]
#[ignore = "loads the in-crate runtime with real base/custom-voice model assets"]
fn base_voice_clone_smoke_generates_pcm_audio() {
    let base_model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Qwen/Qwen3-TTS-12Hz-0.6B-Base");
    let custom_voice_model_dir = custom_voice_model_dir();
    if !base_model_dir.join("config.json").is_file()
        || !custom_voice_model_dir.join("config.json").is_file()
    {
        return;
    }

    let reference_wav = synthesize_reference_wav(&custom_voice_model_dir);
    let engine = load_engine(&base_model_dir);

    let icl_audio = engine
        .synthesize(
            QwenRequest::Base(BaseRequest {
                text: "Hello from the Base voice clone ICL smoke path.".to_string(),
                language: LanguageSelection::Named("English".to_string()),
                voice_clone: Some(BaseVoiceCloneConditioning::ReferenceAudio(
                    BaseVoiceCloneReferenceAudio {
                        path: reference_wav.clone(),
                        transcript: Some("Hello from the generated reference clip.".to_string()),
                        x_vector_only: false,
                    },
                )),
            }),
            smoke_options(),
        )
        .expect("base ICL voice clone should synthesize audio");
    assert_pcm_audio(&icl_audio);

    let xvector_audio = engine
        .synthesize(
            QwenRequest::Base(BaseRequest {
                text: "Hello from the Base voice clone x-vector-only smoke path.".to_string(),
                language: LanguageSelection::Named("English".to_string()),
                voice_clone: Some(BaseVoiceCloneConditioning::ReferenceAudio(
                    BaseVoiceCloneReferenceAudio {
                        path: reference_wav,
                        transcript: None,
                        x_vector_only: true,
                    },
                )),
            }),
            smoke_options(),
        )
        .expect("base x-vector-only clone should synthesize audio");
    assert_pcm_audio(&xvector_audio);
}

fn load_engine(model_dir: &Path) -> Qwen3TtsEngine {
    Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(model_dir.to_path_buf()),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect("real model fixture should load")
}

fn synthesize_reference_wav(custom_voice_model_dir: &Path) -> PathBuf {
    let temp_root =
        std::env::temp_dir().join(format!("tts-rs-base-clone-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&temp_root).unwrap();
    let reference_wav = temp_root.join("base_clone_reference.wav");

    if reference_wav.is_file() {
        return reference_wav;
    }

    let engine = load_engine(custom_voice_model_dir);
    let audio = engine
        .synthesize(
            QwenRequest::CustomVoice(CustomVoiceRequest {
                text: "Hello from the generated reference clip.".to_string(),
                language: LanguageSelection::Named("English".to_string()),
                speaker: Some("Vivian".to_string()),
                instruct: None,
            }),
            smoke_options(),
        )
        .expect("reference custom-voice synthesis should succeed");
    audio.save_wav(&reference_wav).unwrap();
    reference_wav
}

fn smoke_options() -> Qwen3TtsRunOptions {
    Qwen3TtsRunOptions::default()
}

fn assert_pcm_audio(audio: &tts_core::PcmAudio) {
    assert_eq!(audio.sample_rate, 24_000);
    assert_eq!(audio.channels, 1);
    assert!(!audio.pcm_i16.is_empty());
}

fn custom_voice_model_dir() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Qwen");
    [
        root.join("Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        root.join("Qwen3-TTS-12Hz-0___6B-CustomVoice"),
    ]
    .into_iter()
    .find(|path| path.join("config.json").is_file())
    .unwrap_or_else(|| root.join("Qwen3-TTS-12Hz-0.6B-CustomVoice"))
}
