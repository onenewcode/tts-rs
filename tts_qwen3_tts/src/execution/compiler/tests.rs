use std::collections::HashMap;
use std::path::PathBuf;

use super::*;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::Tokenizer;

#[test]
fn load_detects_model_kind_from_package_name() {
    let temp = unique_temp_dir("compiler-unit");
    write_generation_config(&temp.join("generation_config.json"));
    write_model_config(&temp.join("config.json"));
    write_tokenizer_file(&temp.join("tokenizer.json"));

    let base_package = Qwen3TtsPackage {
        package_root: temp.clone(),
        name: "Qwen3-TTS-12Hz-0.6B-Base".to_string(),
        tokenizer_path: temp.join("tokenizer.json"),
        talker_config_path: temp.join("config.json"),
        talker_weights_path: temp.join("weights/talker.safetensors"),
        generation_config: Qwen3TtsGenerationConfigSource::Path(
            temp.join("generation_config.json"),
        ),
        codec_config_path: temp.join("configs/codec.json"),
        codec_weights_path: temp.join("weights/codec.safetensors"),
    };
    let custom_voice_package = Qwen3TtsPackage {
        name: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
        ..base_package.clone()
    };

    let base_compiler = Qwen3TtsRequestCompiler::load(&base_package).unwrap();
    let custom_voice_compiler = Qwen3TtsRequestCompiler::load(&custom_voice_package).unwrap();

    assert!(base_compiler.profiles.base.is_some());
    assert!(base_compiler.profiles.custom_voice.is_none());
    assert!(custom_voice_compiler.profiles.base.is_none());
    assert!(custom_voice_compiler.profiles.custom_voice.is_some());
    assert_eq!(
        custom_voice_compiler
            .profiles
            .custom_voice
            .as_ref()
            .unwrap()
            .control_config
            .spk_id
            .get("chelsie"),
        Some(&4001)
    );
}

#[test]
fn compile_request_uses_base_prompt_and_auto_controls() {
    let compiler = base_compiler_fixture();

    let condition = compiler
        .compile_request(&crate::QwenRequest::Base(crate::BaseRequest::new(
            "hello from the compiler test prompt with many tokens",
        )))
        .unwrap();

    assert!(condition.text_token_ids.len() >= 8);
    assert!(condition.instruct_token_ids.is_none());
    assert_eq!(condition.prompt_recipe, Qwen3TtsPromptRecipe::BasePlain);
    assert_eq!(
        condition.controls.codec_prefix_ids,
        vec![2051, 2052, 2053, 2049, 2048]
    );
    assert_eq!(condition.codec_eos_token_id, 2150);
}

#[test]
fn compile_request_accepts_case_insensitive_language_names() {
    let compiler = base_compiler_fixture();

    let condition = compiler
        .compile_request(&crate::QwenRequest::Base(crate::BaseRequest {
            text: "hello from the compiler test prompt with enough tokens for chinese".to_string(),
            language: crate::LanguageSelection::Named("Chinese".to_string()),
            voice_clone: None,
        }))
        .unwrap();

    assert_eq!(
        condition.controls.codec_prefix_ids,
        vec![2050, 2052, 3001, 2053, 2049, 2048]
    );
}

#[test]
fn compile_request_uses_voice_clone_prompt_recipe_and_tokens() {
    let compiler = base_compiler_fixture();

    let condition = compiler
        .compile_request(&crate::QwenRequest::Base(crate::BaseRequest {
            text: "hello from the compiler clone prompt test".to_string(),
            language: crate::LanguageSelection::Named("Chinese".to_string()),
            voice_clone: Some(crate::BaseVoiceCloneConditioning::Prompt(
                crate::Qwen3TtsVoiceClonePrompt {
                    speaker_embedding: vec![0.1; 1024],
                    ref_codec_token_ids: Some(vec![vec![7101; 16], vec![7102; 16]]),
                    transcript: Some("reference words".to_string()),
                    mode: crate::Qwen3TtsVoiceClonePromptMode::Icl,
                },
            )),
        }))
        .unwrap();

    assert_eq!(
        condition.prompt_recipe,
        Qwen3TtsPromptRecipe::BaseVoiceCloneIcl
    );
    assert_eq!(
        condition.controls.codec_prefix_ids,
        vec![2050, 2052, 3001, 2053, 2049, 2048]
    );
    assert!(condition.voice_clone.is_some());
    assert!(condition
        .voice_clone
        .as_ref()
        .unwrap()
        .ref_text_token_ids
        .as_ref()
        .is_some());
}

#[test]
fn compile_request_includes_custom_voice_speaker_id() {
    let compiler = custom_voice_compiler_fixture();

    let condition = compiler
        .compile_request(&crate::QwenRequest::CustomVoice(
            crate::CustomVoiceRequest {
                text: "ni hao from the compiler test prompt".to_string(),
                language: crate::LanguageSelection::Named("Chinese".to_string()),
                speaker: Some("Chelsie".to_string()),
                instruct: None,
            },
        ))
        .unwrap();

    assert!(condition.text_token_ids.len() >= 8);
    assert!(condition.instruct_token_ids.is_none());
    assert_eq!(
        condition.prompt_recipe,
        Qwen3TtsPromptRecipe::CustomVoicePlain
    );
    assert_eq!(
        condition.controls.codec_prefix_ids,
        vec![2050, 2052, 3001, 2053, 4001, 2049, 2048]
    );
}

#[test]
fn compile_request_marks_custom_voice_instructed_recipe() {
    let compiler = custom_voice_compiler_fixture();

    let condition = compiler
        .compile_request(&crate::QwenRequest::CustomVoice(
            crate::CustomVoiceRequest {
                text: "ni hao from the compiler instructed prompt".to_string(),
                language: crate::LanguageSelection::Auto,
                speaker: Some("Chelsie".to_string()),
                instruct: Some("请用很生气的语气".to_string()),
            },
        ))
        .unwrap();

    assert_eq!(
        condition.prompt_recipe,
        Qwen3TtsPromptRecipe::CustomVoiceInstructed
    );
    assert!(condition.instruct_token_ids.is_some());
    assert!(condition.text_token_ids.len() >= 8);
    assert_ne!(
        condition.text_token_ids,
        condition.instruct_token_ids.clone().unwrap()
    );
}

#[test]
fn compile_request_rejects_unsupported_language() {
    let compiler = base_compiler_fixture();

    let error = compiler
        .compile_request(&crate::QwenRequest::Base(crate::BaseRequest {
            text: "hello".to_string(),
            language: crate::LanguageSelection::Named("fr".to_string()),
            voice_clone: None,
        }))
        .unwrap_err();

    assert!(error.to_string().contains("unsupported language"));
}

#[test]
fn compile_request_rejects_missing_custom_voice_speaker() {
    let compiler = custom_voice_compiler_fixture();

    let error = compiler
        .compile_request(&crate::QwenRequest::CustomVoice(
            crate::CustomVoiceRequest::new("hello"),
        ))
        .unwrap_err();

    assert!(error.to_string().contains("require a non-empty speaker"));
}

#[test]
fn compile_request_rejects_profile_mismatch() {
    let base_compiler = base_compiler_fixture();
    let custom_voice_compiler = custom_voice_compiler_fixture();

    let base_error = base_compiler
        .compile_request(&crate::QwenRequest::CustomVoice(
            crate::CustomVoiceRequest {
                text: "hello".to_string(),
                language: crate::LanguageSelection::Auto,
                speaker: Some("Chelsie".to_string()),
                instruct: None,
            },
        ))
        .unwrap_err();
    assert!(base_error
        .to_string()
        .contains("does not support custom-voice"));

    let custom_voice_error = custom_voice_compiler
        .compile_request(&crate::QwenRequest::Base(crate::BaseRequest::new("hello")))
        .unwrap_err();
    assert!(custom_voice_error
        .to_string()
        .contains("does not support base"));
}

fn unique_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tts-rs-qwen3-compiler-unit-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ))
}

fn base_compiler_fixture() -> Qwen3TtsRequestCompiler {
    compiler_fixture("Qwen3-TTS-12Hz-0.6B-Base")
}

fn custom_voice_compiler_fixture() -> Qwen3TtsRequestCompiler {
    compiler_fixture("Qwen3-TTS-12Hz-0.6B-CustomVoice")
}

fn compiler_fixture(name: &str) -> Qwen3TtsRequestCompiler {
    let temp = unique_temp_dir("compiler-fixture");
    write_generation_config(&temp.join("generation_config.json"));
    write_model_config(&temp.join("config.json"));
    write_tokenizer_file(&temp.join("tokenizer.json"));
    Qwen3TtsRequestCompiler::load(&Qwen3TtsPackage {
        package_root: temp.clone(),
        name: name.to_string(),
        tokenizer_path: temp.join("tokenizer.json"),
        talker_config_path: temp.join("config.json"),
        talker_weights_path: temp.join("model.safetensors"),
        generation_config: Qwen3TtsGenerationConfigSource::Path(
            temp.join("generation_config.json"),
        ),
        codec_config_path: temp.join("speech_tokenizer/config.json"),
        codec_weights_path: temp.join("speech_tokenizer/model.safetensors"),
    })
    .unwrap()
}

fn write_generation_config(path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, GENERATION_CONFIG_JSON).unwrap();
}

fn write_model_config(path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, MODEL_CONFIG_JSON).unwrap();
}

fn write_tokenizer_file(path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_vec(&test_tokenizer()).unwrap()).unwrap();
}

fn test_tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            HashMap::from([(String::from("<unk>"), 0)])
                .into_iter()
                .collect(),
        )
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
    "codec_language_id": {"chinese": 3001, "english": 3002},
    "spk_id": {"chelsie": 4001}
  }
}"#;
