use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{
    LanguageSelection, Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsPackage,
    Qwen3TtsProfilePackage, QwenRequest,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsRequestCompiler {
    pub(crate) profiles: Qwen3TtsCompiledProfiles,
}

impl Qwen3TtsRequestCompiler {
    pub(crate) fn load(package: &Qwen3TtsPackage) -> Result<Self, Qwen3TtsLoadError> {
        Ok(Self {
            profiles: Qwen3TtsCompiledProfiles {
                base: package
                    .profiles
                    .base
                    .as_ref()
                    .map(|profile| load_profile(profile, Qwen3TtsPromptRecipe::Base))
                    .transpose()?,
                custom_voice: package
                    .profiles
                    .custom_voice
                    .as_ref()
                    .map(|profile| load_profile(profile, Qwen3TtsPromptRecipe::CustomVoice))
                    .transpose()?,
            },
        })
    }

    pub(crate) fn compile_request(
        &self,
        request: &QwenRequest,
    ) -> Result<SemanticRequestCondition, Qwen3TtsInferenceError> {
        match request {
            QwenRequest::Base(request) => {
                let profile = self.profiles.base.as_ref().ok_or_else(|| {
                    Qwen3TtsInferenceError::InvalidInput {
                        message: "package does not support base profile".to_string(),
                    }
                })?;
                Ok(SemanticRequestCondition {
                    prompt: build_prompt(&request.text),
                    controls: resolve_base_control_ids(&profile.control_config, &request.language)?,
                })
            }
            QwenRequest::CustomVoice(request) => {
                let profile = self.profiles.custom_voice.as_ref().ok_or_else(|| {
                    Qwen3TtsInferenceError::InvalidInput {
                        message: "package does not support custom_voice profile".to_string(),
                    }
                })?;
                Ok(SemanticRequestCondition {
                    prompt: build_prompt(&request.text),
                    controls: resolve_custom_voice_control_ids(
                        &profile.control_config,
                        &request.language,
                        request.speaker.as_deref(),
                    )?,
                })
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsCompiledProfiles {
    pub(crate) base: Option<Qwen3TtsCompiledProfile>,
    pub(crate) custom_voice: Option<Qwen3TtsCompiledProfile>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsCompiledProfile {
    pub(crate) generation_config: GenerationConfig,
    pub(crate) control_config: Qwen3TtsControlConfig,
    pub(crate) prompt_recipe: Qwen3TtsPromptRecipe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3TtsPromptRecipe {
    Base,
    CustomVoice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemanticRequestCondition {
    pub(crate) prompt: String,
    pub(crate) controls: ProfileControlIds,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileControlIds {
    pub(crate) tts_bos_token_id: i64,
    pub(crate) tts_eos_token_id: i64,
    pub(crate) tts_pad_token_id: i64,
    pub(crate) codec_bos_id: i64,
    pub(crate) codec_pad_id: i64,
    pub(crate) codec_prefix_ids: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(crate) struct GenerationConfig {
    pub(crate) do_sample: bool,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: usize,
    pub(crate) max_new_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct Qwen3TtsControlConfig {
    pub(crate) tts_bos_token_id: i64,
    pub(crate) tts_eos_token_id: i64,
    pub(crate) tts_pad_token_id: i64,
    pub(crate) codec_bos_id: i64,
    pub(crate) codec_eos_token_id: i64,
    pub(crate) codec_pad_id: i64,
    pub(crate) codec_think_id: i64,
    pub(crate) codec_nothink_id: i64,
    pub(crate) codec_think_bos_id: i64,
    pub(crate) codec_think_eos_id: i64,
    #[serde(default)]
    pub(crate) codec_language_id: BTreeMap<String, i64>,
    #[serde(default)]
    pub(crate) spk_id: BTreeMap<String, i64>,
    #[serde(default)]
    pub(crate) spk_is_dialect: BTreeMap<String, DialectFlag>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum DialectFlag {
    Bool(bool),
    Dialect(String),
}

fn load_profile(
    profile: &Qwen3TtsProfilePackage,
    prompt_recipe: Qwen3TtsPromptRecipe,
) -> Result<Qwen3TtsCompiledProfile, Qwen3TtsLoadError> {
    Ok(Qwen3TtsCompiledProfile {
        generation_config: load_json(&profile.generation_config_path)?,
        control_config: load_json(&profile.control_config_path)?,
        prompt_recipe,
    })
}

fn load_json<T>(path: &Path) -> Result<T, Qwen3TtsLoadError>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = std::fs::read_to_string(path).map_err(|source| Qwen3TtsLoadError::CompilerConfigIo {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| Qwen3TtsLoadError::CompilerConfigParse {
        path: path.to_path_buf(),
        source,
    })
}

fn build_prompt(text: &str) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        text
    )
}

fn resolve_base_control_ids(
    config: &Qwen3TtsControlConfig,
    language: &LanguageSelection,
) -> Result<ProfileControlIds, Qwen3TtsInferenceError> {
    let codec_prefix_ids = match language_key(language)? {
        None => vec![
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
            config.codec_pad_id,
            config.codec_bos_id,
        ],
        Some(language) => vec![
            config.codec_think_id,
            config.codec_think_bos_id,
            lookup_language_id(config, &language)?,
            config.codec_think_eos_id,
            config.codec_pad_id,
            config.codec_bos_id,
        ],
    };

    Ok(ProfileControlIds {
        tts_bos_token_id: config.tts_bos_token_id,
        tts_eos_token_id: config.tts_eos_token_id,
        tts_pad_token_id: config.tts_pad_token_id,
        codec_bos_id: config.codec_bos_id,
        codec_pad_id: config.codec_pad_id,
        codec_prefix_ids,
    })
}

fn resolve_custom_voice_control_ids(
    config: &Qwen3TtsControlConfig,
    language: &LanguageSelection,
    speaker: Option<&str>,
) -> Result<ProfileControlIds, Qwen3TtsInferenceError> {
    let speaker = speaker.map(normalize_key);
    let mut language = language_key(language)?;

    if matches!(language.as_deref(), None | Some("chinese" | "zh")) {
        if let Some(speaker_name) = speaker.as_deref() {
            if let Some(DialectFlag::Dialect(dialect)) = config.spk_is_dialect.get(speaker_name) {
                language = Some(normalize_key(dialect));
            }
        }
    }

    let mut codec_prefix_ids = match language {
        Some(language) => vec![
            config.codec_think_id,
            config.codec_think_bos_id,
            lookup_language_id(config, &language)?,
            config.codec_think_eos_id,
        ],
        None => vec![
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ],
    };

    if let Some(speaker_name) = speaker.as_deref().filter(|speaker_name| !speaker_name.is_empty()) {
        codec_prefix_ids.push(lookup_speaker_id(config, speaker_name)?);
    }
    codec_prefix_ids.extend([config.codec_pad_id, config.codec_bos_id]);

    Ok(ProfileControlIds {
        tts_bos_token_id: config.tts_bos_token_id,
        tts_eos_token_id: config.tts_eos_token_id,
        tts_pad_token_id: config.tts_pad_token_id,
        codec_bos_id: config.codec_bos_id,
        codec_pad_id: config.codec_pad_id,
        codec_prefix_ids,
    })
}

fn language_key(language: &LanguageSelection) -> Result<Option<String>, Qwen3TtsInferenceError> {
    match language {
        LanguageSelection::Auto => Ok(None),
        LanguageSelection::Named(language) => {
            let normalized = normalize_key(language);
            if normalized.is_empty() {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: "language cannot be empty".to_string(),
                });
            }
            Ok(Some(normalized))
        }
    }
}

fn lookup_language_id(
    config: &Qwen3TtsControlConfig,
    language: &str,
) -> Result<i64, Qwen3TtsInferenceError> {
    config
        .codec_language_id
        .get(language)
        .copied()
        .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
            message: format!("unsupported language: {language}"),
        })
}

fn lookup_speaker_id(
    config: &Qwen3TtsControlConfig,
    speaker: &str,
) -> Result<i64, Qwen3TtsInferenceError> {
    config
        .spk_id
        .get(speaker)
        .copied()
        .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
            message: format!("unsupported speaker: {speaker}"),
        })
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{Qwen3TtsPackageProfiles, Qwen3TtsProfilePackage};

    #[test]
    fn load_uses_fixed_profile_fields_and_prompt_recipes() {
        let temp = unique_temp_dir("compiler-unit");
        std::fs::create_dir_all(temp.join("profiles/base")).unwrap();
        std::fs::create_dir_all(temp.join("profiles/custom_voice")).unwrap();
        write_profile_files(&temp.join("profiles/base"));
        write_profile_files(&temp.join("profiles/custom_voice"));

        let package = Qwen3TtsPackage {
            package_root: temp.clone(),
            name: "unit-fixture".to_string(),
            tokenizer_path: temp.join("tokenizer.json"),
            talker_config_path: temp.join("configs/talker.json"),
            talker_weights_path: temp.join("weights/talker.safetensors"),
            codec_config_path: temp.join("configs/codec.json"),
            codec_weights_path: temp.join("weights/codec.safetensors"),
            profiles: Qwen3TtsPackageProfiles {
                base: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/base/generation_config.json"),
                    control_config_path: temp.join("profiles/base/control_config.json"),
                }),
                custom_voice: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/custom_voice/generation_config.json"),
                    control_config_path: temp.join("profiles/custom_voice/control_config.json"),
                }),
            },
        };

        let compiler = Qwen3TtsRequestCompiler::load(&package).unwrap();

        assert_eq!(
            compiler.profiles.base.as_ref().unwrap().prompt_recipe,
            Qwen3TtsPromptRecipe::Base
        );
        assert_eq!(
            compiler
                .profiles
                .custom_voice
                .as_ref()
                .unwrap()
                .prompt_recipe,
            Qwen3TtsPromptRecipe::CustomVoice
        );
        assert_eq!(
            compiler
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
        let compiler = compiler_fixture();

        let condition = compiler
            .compile_request(&crate::QwenRequest::Base(crate::BaseRequest::new("hello")))
            .unwrap();

        assert_eq!(
            condition.prompt,
            "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            condition.controls.codec_prefix_ids,
            vec![2051, 2052, 2053, 2049, 2048]
        );
    }

    #[test]
    fn compile_request_resolves_custom_voice_speaker_dialect() {
        let compiler = compiler_fixture();

        let condition = compiler
            .compile_request(&crate::QwenRequest::CustomVoice(crate::CustomVoiceRequest {
                text: "ni hao".to_string(),
                language: crate::LanguageSelection::Auto,
                speaker: Some("Chelsie".to_string()),
            }))
            .unwrap();

        assert_eq!(
            condition.controls.codec_prefix_ids,
            vec![2050, 2052, 3001, 2053, 4001, 2049, 2048]
        );
    }

    #[test]
    fn compile_request_rejects_unsupported_language() {
        let compiler = compiler_fixture();

        let error = compiler
            .compile_request(&crate::QwenRequest::Base(crate::BaseRequest {
                text: "hello".to_string(),
                language: crate::LanguageSelection::Named("fr".to_string()),
            }))
            .unwrap_err();

        assert!(error.to_string().contains("unsupported language"));
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

    fn compiler_fixture() -> Qwen3TtsRequestCompiler {
        let temp = unique_temp_dir("compiler-fixture");
        std::fs::create_dir_all(temp.join("profiles/base")).unwrap();
        std::fs::create_dir_all(temp.join("profiles/custom_voice")).unwrap();
        write_profile_files(&temp.join("profiles/base"));
        write_profile_files(&temp.join("profiles/custom_voice"));
        Qwen3TtsRequestCompiler::load(&Qwen3TtsPackage {
            package_root: temp.clone(),
            name: "fixture".to_string(),
            tokenizer_path: temp.join("tokenizer.json"),
            talker_config_path: temp.join("configs/talker.json"),
            talker_weights_path: temp.join("weights/talker.safetensors"),
            codec_config_path: temp.join("configs/codec.json"),
            codec_weights_path: temp.join("weights/codec.safetensors"),
            profiles: Qwen3TtsPackageProfiles {
                base: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/base/generation_config.json"),
                    control_config_path: temp.join("profiles/base/control_config.json"),
                }),
                custom_voice: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/custom_voice/generation_config.json"),
                    control_config_path: temp.join("profiles/custom_voice/control_config.json"),
                }),
            },
        })
        .unwrap()
    }

    fn write_profile_files(dir: &Path) {
        std::fs::write(dir.join("generation_config.json"), GENERATION_CONFIG_JSON).unwrap();
        std::fs::write(dir.join("control_config.json"), CONTROL_CONFIG_JSON).unwrap();
    }

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
}
