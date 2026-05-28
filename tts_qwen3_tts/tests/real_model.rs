#![cfg(feature = "flex")]

use std::fs;
use std::path::{Path, PathBuf};

use tts_qwen3_tts::{
    BaseRequest, Qwen3TtsBackend, Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsPackageSource,
    Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

#[test]
#[ignore = "loads the in-crate runtime with real model assets"]
fn engine_synthesizes_real_audio_with_in_crate_runtime() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice");
    if !model_dir.join("config.json").is_file() {
        return;
    }

    let manifest_path = write_manifest_fixture(&model_dir);
    let engine = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ManifestPath(manifest_path),
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

fn write_manifest_fixture(model_dir: &Path) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-real-runtime-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("profiles/base")).unwrap();
    fs::create_dir_all(package_dir.join("profiles/custom_voice")).unwrap();
    fs::write(package_dir.join("profiles/base/control_config.json"), CONTROL_CONFIG_JSON).unwrap();
    fs::write(
        package_dir.join("profiles/custom_voice/control_config.json"),
        CONTROL_CONFIG_JSON,
    )
    .unwrap();

    let manifest = format!(
        "format: qwen3_tts_package/v1\nname: qwen3-tts-12hz-0.6b-customvoice\n\nartifacts:\n  tokenizer: {tokenizer}\n  talker_config: {talker_config}\n  talker_weights: {talker_weights}\n  codec_config: {codec_config}\n  codec_weights: {codec_weights}\n\nprofiles:\n  base:\n    generation_config: {generation_config}\n    control_config: {base_control}\n  custom_voice:\n    generation_config: {generation_config}\n    control_config: {custom_control}\n",
        tokenizer = yaml_path(&model_dir.join("vocab.json")),
        talker_config = yaml_path(&model_dir.join("config.json")),
        talker_weights = yaml_path(&model_dir.join("model.safetensors")),
        codec_config = yaml_path(&model_dir.join("speech_tokenizer/config.json")),
        codec_weights = yaml_path(&model_dir.join("speech_tokenizer/model.safetensors")),
        generation_config = yaml_path(&model_dir.join("generation_config.json")),
        base_control = yaml_path(&package_dir.join("profiles/base/control_config.json")),
        custom_control = yaml_path(&package_dir.join("profiles/custom_voice/control_config.json")),
    );
    let manifest_path = package_dir.join("qwen3_tts_package.yaml");
    fs::write(&manifest_path, manifest).unwrap();
    manifest_path
}

fn yaml_path(path: &Path) -> String {
    path.canonicalize().unwrap().display().to_string()
}

const CONTROL_CONFIG_JSON: &str = r#"{
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "codec_bos_id": 2048,
  "codec_eos_token_id": 2150,
  "codec_pad_id": 2049,
  "codec_think_id": 2050,
  "codec_nothink_id": 2051,
  "codec_think_bos_id": 2052,
  "codec_think_eos_id": 2053,
  "codec_language_id": {"zh": 3001, "chinese": 3001},
  "spk_id": {"chelsie": 4001},
  "spk_is_dialect": {"chelsie": "zh"}
}"#;
