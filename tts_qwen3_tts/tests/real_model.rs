#![cfg(feature = "flex")]

use std::path::PathBuf;

use tts_qwen3_tts::{
    BaseRequest, Qwen3TtsBackend, Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsPackageSource,
    Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

#[test]
#[ignore = "loads the in-crate runtime with real model assets"]
fn engine_synthesizes_real_audio_with_in_crate_runtime() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice");
    if !model_dir.join("config.json").is_file() {
        return;
    }

    let engine = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(model_dir),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect("real model fixture should load");

    let audio = engine
        .synthesize(
            QwenRequest::Base(BaseRequest::new("你好，欢迎使用语音合成。")),
            Qwen3TtsRunOptions {
                max_new_tokens: 4,
                ..Qwen3TtsRunOptions::default()
            },
        )
        .expect("in-crate runtime should synthesize audio");

    assert_eq!(audio.sample_rate, 24_000);
    assert_eq!(audio.channels, 1);
    assert!(!audio.pcm_i16.is_empty());
}
