use std::fs;
use std::path::{Path, PathBuf};

use tts_qwen3_tts::{
    BaseRequest, Qwen3TtsBackend, Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsError,
    Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

#[test]
fn synthesize_routes_through_infer_engine_backbone() {
    let package_dir = write_package_fixture("engine-backbone");
    let engine = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::PackageDir(package_dir),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect("package fixture should load");

    let error = engine
        .synthesize(
            QwenRequest::Base(BaseRequest::new("hello")),
            Qwen3TtsRunOptions::default(),
        )
        .expect_err("runtime port is not finished yet");

    assert!(matches!(error, Qwen3TtsError::Infer(_)));
}

fn write_package_fixture(label: &str) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-engine-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("profiles/base")).unwrap();
    fs::create_dir_all(package_dir.join("profiles/custom_voice")).unwrap();
    fs::write(package_dir.join("qwen3_tts_package.yaml"), PACKAGE_YAML).unwrap();
    write_profile_files(&package_dir.join("profiles/base"));
    write_profile_files(&package_dir.join("profiles/custom_voice"));
    package_dir
}

fn write_profile_files(dir: &Path) {
    fs::write(dir.join("generation_config.json"), GENERATION_CONFIG_JSON).unwrap();
    fs::write(dir.join("control_config.json"), CONTROL_CONFIG_JSON).unwrap();
}

const PACKAGE_YAML: &str = r#"format: qwen3_tts_package/v1
name: engine-fixture

artifacts:
  tokenizer: tokenizer.json
  talker_config: configs/talker.json
  talker_weights: weights/talker.safetensors
  codec_config: configs/codec.json
  codec_weights: weights/codec.safetensors

profiles:
  base:
    generation_config: profiles/base/generation_config.json
    control_config: profiles/base/control_config.json
  custom_voice:
    generation_config: profiles/custom_voice/generation_config.json
    control_config: profiles/custom_voice/control_config.json
"#;

const GENERATION_CONFIG_JSON: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_new_tokens": 8192
}"#;

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
  "codec_language_id": {"zh": 3001},
  "spk_id": {"chelsie": 4001},
  "spk_is_dialect": {"chelsie": "zh"}
}"#;
