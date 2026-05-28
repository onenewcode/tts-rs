use std::fs;
use std::path::{Path, PathBuf};

use tts_qwen3_tts::{
    BaseRequest, CustomVoiceRequest, LanguageSelection, Qwen3TtsBackend, Qwen3TtsEngine,
    Qwen3TtsEngineConfig, Qwen3TtsPackage, Qwen3TtsPackageSource, Qwen3TtsProfilingConfig,
    Qwen3TtsRunOptions, QwenRequest, SamplingConfig,
};

#[test]
fn base_request_defaults_language_to_auto() {
    let request = BaseRequest::new("hello");
    assert_eq!(request.text, "hello");
    assert_eq!(request.language, LanguageSelection::Auto);
}

#[test]
fn custom_voice_request_defaults_to_auto_language_and_no_speaker() {
    let request = CustomVoiceRequest::new("hello");
    assert_eq!(request.text, "hello");
    assert_eq!(request.language, LanguageSelection::Auto);
    assert_eq!(request.speaker, None);
}

#[test]
fn run_options_default_to_greedy_256_tokens() {
    let options = Qwen3TtsRunOptions::default();
    assert_eq!(options.max_new_tokens, 256);
    assert_eq!(options.sampling, SamplingConfig::greedy());
}

#[test]
fn profiling_defaults_match_refactor_contract() {
    let profiling = Qwen3TtsProfilingConfig::default();
    assert!(!profiling.enabled);
    assert!(!profiling.per_step);
    assert!(profiling.stage_summary);
    assert_eq!(profiling.log_topk, 8);
}

#[test]
fn backend_parses_supported_labels() {
    assert_eq!("flex".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Flex);
    assert_eq!("wgpu".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Wgpu);
    assert_eq!("cuda".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Cuda);
    assert_eq!("rocm".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Rocm);
    assert_eq!("metal".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Metal);
    assert_eq!("vulkan".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::Vulkan);
    assert_eq!("webgpu".parse::<Qwen3TtsBackend>().unwrap(), Qwen3TtsBackend::WebGpu);
}

#[test]
fn engine_load_normalizes_manifest_relative_paths() {
    let package_dir = write_package_fixture("package-load");
    let engine = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::PackageDir(package_dir.clone()),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect("package fixture should load");

    let package = engine.package();
    assert_eq!(package.name, "fixture-package");
    assert_eq!(package.package_root, package_dir);
    assert_eq!(package.tokenizer_path, package.package_root.join("tokenizer.json"));
    assert_eq!(
        package
            .profiles
            .custom_voice
            .as_ref()
            .expect("custom voice profile should exist")
            .control_config_path,
        package.package_root.join("profiles/custom_voice/control_config.json")
    );
}

#[test]
fn package_load_rejects_unknown_manifest_format() {
    let package_dir = write_package_fixture("bad-format");
    fs::write(
        package_dir.join("qwen3_tts_package.yaml"),
        PACKAGE_YAML.replace("qwen3_tts_package/v1", "wrong/v1"),
    )
    .unwrap();

    let error = Qwen3TtsPackage::load(&Qwen3TtsPackageSource::PackageDir(package_dir))
        .expect_err("invalid format should fail");

    assert!(error.to_string().contains("unsupported package format"));
}

#[test]
fn request_enum_preserves_profile_specific_payloads() {
    let request = QwenRequest::CustomVoice(CustomVoiceRequest {
        text: "hello".to_string(),
        language: LanguageSelection::Named("zh".to_string()),
        speaker: Some("Chelsie".to_string()),
    });

    match request {
        QwenRequest::CustomVoice(inner) => {
            assert_eq!(inner.language, LanguageSelection::Named("zh".to_string()));
            assert_eq!(inner.speaker.as_deref(), Some("Chelsie"));
        }
        QwenRequest::Base(_) => panic!("expected custom voice request"),
    }
}

fn write_package_fixture(label: &str) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-package-{label}-{}",
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
name: fixture-package

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
