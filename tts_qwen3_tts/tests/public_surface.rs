use std::fs;
use std::path::PathBuf;

use tts_qwen3_tts::{
    BaseRequest, CustomVoiceRequest, LanguageSelection, Qwen3TtsBackend,
    Qwen3TtsGenerationConfigSource, Qwen3TtsPackage, Qwen3TtsPackageSource,
    Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest, SamplingConfig,
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
    assert_eq!(
        "flex".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Flex
    );
    assert_eq!(
        "wgpu".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Wgpu
    );
    assert_eq!(
        "cuda".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Cuda
    );
    assert_eq!(
        "rocm".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Rocm
    );
    assert_eq!(
        "metal".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Metal
    );
    assert_eq!(
        "vulkan".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::Vulkan
    );
    assert_eq!(
        "webgpu".parse::<Qwen3TtsBackend>().unwrap(),
        Qwen3TtsBackend::WebGpu
    );
}

#[test]
fn engine_load_normalizes_model_dir_paths() {
    let model_dir = write_model_dir_fixture("model-dir-load");
    let package = Qwen3TtsPackage::load(&Qwen3TtsPackageSource::ModelDir(model_dir.clone()))
        .expect("model-dir fixture should normalize hub-style paths");
    assert_eq!(
        package.name,
        model_dir.file_name().unwrap().to_string_lossy()
    );
    assert_eq!(package.package_root, model_dir);
    assert_eq!(
        package.tokenizer_path,
        package.package_root.join("vocab.json")
    );
    assert!(matches!(
        package.generation_config,
        Qwen3TtsGenerationConfigSource::Path(_)
    ));
}

#[test]
fn package_load_rejects_unknown_manifest_format() {
    let package_dir = write_manifest_fixture("bad-format");
    let manifest_path = package_dir.join("qwen3_tts_package.yaml");
    fs::write(
        &manifest_path,
        PACKAGE_YAML.replace("qwen3_tts_package/v1", "wrong/v1"),
    )
    .unwrap();

    let error = Qwen3TtsPackage::load(&Qwen3TtsPackageSource::ManifestPath(manifest_path))
        .expect_err("invalid format should fail");

    assert!(error.to_string().contains("unsupported package format"));
}

#[test]
fn manifest_load_normalizes_relative_paths() {
    let package_dir = write_manifest_fixture("manifest-load");
    let manifest_path = package_dir.join("qwen3_tts_package.yaml");
    let package = Qwen3TtsPackage::load(&Qwen3TtsPackageSource::ManifestPath(manifest_path))
        .expect("manifest fixture should normalize relative paths");

    assert_eq!(package.name, "fixture-package");
    assert_eq!(package.package_root, package_dir);
    assert_eq!(
        package.tokenizer_path,
        package.package_root.join("tokenizer.json")
    );
    assert!(matches!(
        package.generation_config,
        Qwen3TtsGenerationConfigSource::Inline(_)
    ));
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

fn write_model_dir_fixture(label: &str) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-model-dir-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("speech_tokenizer")).unwrap();
    write_file(package_dir.join("vocab.json"), "{}");
    write_file(package_dir.join("merges.txt"), "");
    write_file(package_dir.join("config.json"), MODEL_CONFIG_JSON);
    write_file(
        package_dir.join("generation_config.json"),
        GENERATION_CONFIG_JSON,
    );
    write_file(package_dir.join("model.safetensors"), "");
    write_file(package_dir.join("speech_tokenizer/config.json"), "{}");
    write_file(package_dir.join("speech_tokenizer/model.safetensors"), "");
    package_dir
}

fn write_manifest_fixture(label: &str) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-manifest-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("configs")).unwrap();
    fs::create_dir_all(package_dir.join("weights")).unwrap();
    write_file(package_dir.join("qwen3_tts_package.yaml"), PACKAGE_YAML);
    write_file(package_dir.join("tokenizer.json"), "{}");
    write_file(package_dir.join("configs/talker.json"), MODEL_CONFIG_JSON);
    write_file(package_dir.join("weights/talker.safetensors"), "");
    write_file(package_dir.join("configs/codec.json"), "{}");
    write_file(package_dir.join("weights/codec.safetensors"), "");
    package_dir
}

fn write_file(path: PathBuf, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

const PACKAGE_YAML: &str = r#"format: qwen3_tts_package/v1
name: fixture-package

artifacts:
  tokenizer: tokenizer.json
  talker_config: configs/talker.json
  talker_weights: weights/talker.safetensors
  codec_config: configs/codec.json
  codec_weights: weights/codec.safetensors

generation_config:
  do_sample: true
  repetition_penalty: 1.05
  temperature: 0.9
  top_p: 1.0
  top_k: 50
  max_new_tokens: 8192
"#;

const GENERATION_CONFIG_JSON: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_new_tokens": 8192
}"#;

const MODEL_CONFIG_JSON: &str = r#"{
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "talker_config": {
    "codec_bos_id": 2048,
    "codec_eos_token_id": 2150,
    "codec_pad_id": 2049,
    "codec_think_id": 2050,
    "codec_nothink_id": 2051,
    "codec_think_bos_id": 2052,
    "codec_think_eos_id": 2053,
    "codec_language_id": {"chinese": 3001},
    "spk_id": {"chelsie": 4001}
  }
}"#;
