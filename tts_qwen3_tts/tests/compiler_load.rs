use std::fs;
use std::path::{Path, PathBuf};

use tts_qwen3_tts::{
    Qwen3TtsBackend, Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsPackageSource,
    Qwen3TtsProfilingConfig,
};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

#[test]
fn engine_load_requires_profile_config_files() {
    let package_dir = write_package_fixture("missing-profile-configs", false);

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::PackageDir(package_dir),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect_err("engine load should fail when compiled profile files are absent");

    let message = error.to_string();
    assert!(
        message.contains("generation_config.json") || message.contains("control_config.json"),
        "unexpected error: {message}"
    );
}

#[test]
fn engine_load_requires_runtime_artifacts_after_profile_config_parse() {
    let package_dir = write_package_fixture("present-profile-configs", true);

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::PackageDir(package_dir),
        backend: Qwen3TtsBackend::Flex,
        profiling: Qwen3TtsProfilingConfig::default(),
    })
    .expect_err("engine load should also validate resident runtime artifacts during load");

    let message = error.to_string();
    assert!(
        message.contains("tokenizer")
            || message.contains("talker")
            || message.contains("codec")
            || message.contains("config.json")
            || message.contains("model.safetensors"),
        "unexpected error: {message}"
    );
}

fn write_package_fixture(label: &str, include_profile_files: bool) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-compiler-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("profiles/base")).unwrap();
    fs::create_dir_all(package_dir.join("profiles/custom_voice")).unwrap();
    fs::write(package_dir.join("qwen3_tts_package.yaml"), PACKAGE_YAML).unwrap();
    write_tokenizer_file(&package_dir.join("tokenizer.json"));

    if include_profile_files {
        write_profile_files(&package_dir.join("profiles/base"));
        write_profile_files(&package_dir.join("profiles/custom_voice"));
    }

    package_dir
}

fn write_profile_files(dir: &Path) {
    fs::write(dir.join("generation_config.json"), GENERATION_CONFIG_JSON).unwrap();
    fs::write(dir.join("control_config.json"), CONTROL_CONFIG_JSON).unwrap();
}

fn write_tokenizer_file(path: &Path) {
    fs::write(path, serde_json::to_vec(&test_tokenizer()).unwrap()).unwrap();
}

fn test_tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab([(String::from("<unk>"), 0u32)].into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

const PACKAGE_YAML: &str = r#"format: qwen3_tts_package/v1
name: compiler-fixture

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
