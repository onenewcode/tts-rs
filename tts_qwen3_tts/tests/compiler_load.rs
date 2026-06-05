use std::fs;
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tts_qwen3_tts::{
    Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsPackageSource, Qwen3TtsProfilingConfig,
};

#[test]
fn engine_load_requires_generation_config_file() {
    let model_dir = write_model_dir_fixture("missing-generation-config", false);

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(model_dir),
        profiling: Qwen3TtsProfilingConfig::default(),
        talker_dtype: None,
        codec_dtype: None,
    })
    .expect_err("engine load should fail when generation_config.json is absent");

    let message = error.to_string();
    assert!(
        message.contains("generation_config.json"),
        "unexpected error: {message}"
    );
}

#[test]
fn engine_load_requires_runtime_artifacts_after_model_dir_parse() {
    let model_dir = write_model_dir_fixture("present-generation-config", true);

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(model_dir),
        profiling: Qwen3TtsProfilingConfig::default(),
        talker_dtype: None,
        codec_dtype: None,
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

#[test]
fn engine_load_rejects_zero_max_new_tokens_in_generation_config() {
    let model_dir = write_model_dir_fixture("zero-max-new-tokens", true);
    fs::write(
        model_dir.join("generation_config.json"),
        INVALID_GENERATION_CONFIG_JSON,
    )
    .unwrap();

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(model_dir),
        profiling: Qwen3TtsProfilingConfig::default(),
        talker_dtype: None,
        codec_dtype: None,
    })
    .expect_err("engine load should reject zero generation_config.max_new_tokens");

    let message = error.to_string();
    assert!(
        message.contains("generation_config.max_new_tokens")
            && message.contains("greater than zero"),
        "unexpected error: {message}"
    );
}

#[test]
fn engine_load_rejects_zero_max_new_tokens_in_manifest_generation_config() {
    let manifest_path = write_manifest_fixture("manifest-zero-max-new-tokens", 0);

    let error = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ManifestPath(manifest_path),
        profiling: Qwen3TtsProfilingConfig::default(),
        talker_dtype: None,
        codec_dtype: None,
    })
    .expect_err("engine load should reject zero manifest generation_config.max_new_tokens");

    let message = error.to_string();
    assert!(
        message.contains("generation_config.max_new_tokens")
            && message.contains("greater than zero"),
        "unexpected error: {message}"
    );
}

fn write_model_dir_fixture(label: &str, include_generation_config: bool) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-model-dir-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(package_dir.join("speech_tokenizer")).unwrap();
    write_tokenizer_file(&package_dir.join("tokenizer.json"));
    fs::write(package_dir.join("config.json"), MODEL_CONFIG_JSON).unwrap();
    fs::write(package_dir.join("model.safetensors"), "").unwrap();
    fs::write(package_dir.join("speech_tokenizer/config.json"), "{}").unwrap();
    fs::write(package_dir.join("speech_tokenizer/model.safetensors"), "").unwrap();

    if include_generation_config {
        fs::write(
            package_dir.join("generation_config.json"),
            GENERATION_CONFIG_JSON,
        )
        .unwrap();
    }

    package_dir
}

fn write_manifest_fixture(label: &str, max_new_tokens: usize) -> PathBuf {
    let package_dir = std::env::temp_dir().join(format!(
        "tts-rs-qwen3-manifest-{label}-{}",
        std::process::id()
    ));
    if package_dir.exists() {
        fs::remove_dir_all(&package_dir).unwrap();
    }
    fs::create_dir_all(&package_dir).unwrap();

    let manifest = format!(
        r#"format: qwen3_tts_package/v1
name: Qwen3-TTS-12Hz-0.6B-Base
artifacts:
  tokenizer: ./tokenizer.json
  talker_config: ./config.json
  talker_weights: ./model.safetensors
  codec_config: ./speech_tokenizer/config.json
  codec_weights: ./speech_tokenizer/model.safetensors
generation_config:
  do_sample: true
  repetition_penalty: 1.05
  temperature: 0.9
  top_p: 1.0
  top_k: 50
  max_new_tokens: {max_new_tokens}
"#
    );

    let manifest_path = package_dir.join("qwen3_tts_package.yaml");
    fs::write(&manifest_path, manifest).unwrap();
    manifest_path
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

const GENERATION_CONFIG_JSON: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_new_tokens": 8192
}"#;

const INVALID_GENERATION_CONFIG_JSON: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_new_tokens": 0
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
