pub(crate) mod session_seed;

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    LanguageSelection, Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource,
    Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsPackage, QwenRequest,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsRequestCompiler {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) profiles: Qwen3TtsCompiledProfiles,
}

impl Qwen3TtsRequestCompiler {
    pub(crate) fn load(package: &Qwen3TtsPackage) -> Result<Self, Qwen3TtsLoadError> {
        let generation_config = load_generation_config(&package.generation_config)?;
        let control_config = load_control_config(&package.talker_config_path)?;

        Ok(Self {
            tokenizer: crate::io::tokenizer::load_qwen3_tts_tokenizer(&package.tokenizer_path)
                .map_err(|source| Qwen3TtsLoadError::Tokenizer {
                    path: package.tokenizer_path.clone(),
                    source,
                })?,
            profiles: Qwen3TtsCompiledProfiles {
                base: Some(Qwen3TtsCompiledProfile {
                    generation_config: generation_config.clone(),
                    control_config: control_config.clone(),
                    prompt_recipe: Qwen3TtsPromptRecipe::Base,
                }),
                custom_voice: if control_config.spk_id.is_empty() {
                    None
                } else {
                    Some(Qwen3TtsCompiledProfile {
                        generation_config,
                        control_config,
                        prompt_recipe: Qwen3TtsPromptRecipe::CustomVoice,
                    })
                },
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
                compile_profile_condition(
                    &self.tokenizer,
                    &build_prompt(&request.text),
                    resolve_base_control_ids(&profile.control_config, &request.language)?,
                    profile.control_config.codec_eos_token_id as usize,
                )
            }
            QwenRequest::CustomVoice(request) => {
                let profile = self.profiles.custom_voice.as_ref().ok_or_else(|| {
                    Qwen3TtsInferenceError::InvalidInput {
                        message: "model does not support custom-voice requests; no speakers were found in config.json".to_string(),
                    }
                })?;
                compile_profile_condition(
                    &self.tokenizer,
                    &build_prompt(&request.text),
                    resolve_custom_voice_control_ids(
                        &profile.control_config,
                        &request.language,
                        request.speaker.as_deref(),
                    )?,
                    profile.control_config.codec_eos_token_id as usize,
                )
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
    pub(crate) text_token_ids: Vec<i64>,
    pub(crate) controls: ProfileControlIds,
    pub(crate) codec_eos_token_id: usize,
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

impl From<Qwen3TtsGenerationConfigManifest> for GenerationConfig {
    fn from(value: Qwen3TtsGenerationConfigManifest) -> Self {
        Self {
            do_sample: value.do_sample,
            repetition_penalty: value.repetition_penalty,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            max_new_tokens: value.max_new_tokens,
        }
    }
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
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct Qwen3TtsControlConfigFile {
    tts_bos_token_id: i64,
    tts_eos_token_id: i64,
    tts_pad_token_id: i64,
    talker_config: Qwen3TtsControlConfigBody,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct Qwen3TtsControlConfigBody {
    codec_bos_id: i64,
    codec_eos_token_id: i64,
    codec_pad_id: i64,
    codec_think_id: i64,
    codec_nothink_id: i64,
    codec_think_bos_id: i64,
    codec_think_eos_id: i64,
    #[serde(default)]
    codec_language_id: BTreeMap<String, i64>,
    #[serde(default)]
    spk_id: BTreeMap<String, i64>,
}

fn load_control_config(path: &Path) -> Result<Qwen3TtsControlConfig, Qwen3TtsLoadError> {
    let raw: Qwen3TtsControlConfigFile = load_json(path)?;
    Ok(Qwen3TtsControlConfig {
        tts_bos_token_id: raw.tts_bos_token_id,
        tts_eos_token_id: raw.tts_eos_token_id,
        tts_pad_token_id: raw.tts_pad_token_id,
        codec_bos_id: raw.talker_config.codec_bos_id,
        codec_eos_token_id: raw.talker_config.codec_eos_token_id,
        codec_pad_id: raw.talker_config.codec_pad_id,
        codec_think_id: raw.talker_config.codec_think_id,
        codec_nothink_id: raw.talker_config.codec_nothink_id,
        codec_think_bos_id: raw.talker_config.codec_think_bos_id,
        codec_think_eos_id: raw.talker_config.codec_think_eos_id,
        codec_language_id: normalize_language_map(raw.talker_config.codec_language_id),
        spk_id: normalize_key_map(raw.talker_config.spk_id),
    })
}

fn load_generation_config(
    source: &Qwen3TtsGenerationConfigSource,
) -> Result<GenerationConfig, Qwen3TtsLoadError> {
    match source {
        Qwen3TtsGenerationConfigSource::Path(path) => load_json(path),
        Qwen3TtsGenerationConfigSource::Inline(config) => {
            Ok(GenerationConfig::from(config.clone()))
        }
    }
}

fn load_json<T>(path: &Path) -> Result<T, Qwen3TtsLoadError>
where
    T: for<'de> Deserialize<'de>,
{
    let raw =
        std::fs::read_to_string(path).map_err(|source| Qwen3TtsLoadError::CompilerConfigIo {
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

fn compile_profile_condition(
    tokenizer: &Tokenizer,
    prompt: &str,
    controls: ProfileControlIds,
    codec_eos_token_id: usize,
) -> Result<SemanticRequestCondition, Qwen3TtsInferenceError> {
    let text_token_ids = tokenizer
        .encode(prompt, false)
        .map_err(|source| Qwen3TtsInferenceError::Tokenizer { source })?
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    if text_token_ids.len() < 8 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "qwen prompt tokenization is too short: {} tokens",
                text_token_ids.len()
            ),
        });
    }

    Ok(SemanticRequestCondition {
        text_token_ids,
        controls,
        codec_eos_token_id,
    })
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
    let language = language_key(language)?;

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

    if let Some(speaker_name) = speaker
        .as_deref()
        .filter(|speaker_name| !speaker_name.is_empty())
    {
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
            message: format!(
                "unsupported language: {language}; supported languages: {}",
                config
                    .codec_language_id
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
            message: format!(
                "unsupported speaker: {speaker}; supported speakers: {}",
                config
                    .spk_id
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn normalize_language_map(map: BTreeMap<String, i64>) -> BTreeMap<String, i64> {
    map.into_iter()
        .map(|(key, value)| (normalize_key(&key), value))
        .collect()
}

fn normalize_key_map(map: BTreeMap<String, i64>) -> BTreeMap<String, i64> {
    map.into_iter()
        .map(|(key, value)| (normalize_key(&key), value))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    #[test]
    fn load_uses_fixed_profile_fields_and_prompt_recipes() {
        let temp = unique_temp_dir("compiler-unit");
        write_generation_config(&temp.join("generation_config.json"));
        write_model_config(&temp.join("config.json"));
        write_tokenizer_file(&temp.join("tokenizer.json"));

        let package = Qwen3TtsPackage {
            package_root: temp.clone(),
            name: "unit-fixture".to_string(),
            tokenizer_path: temp.join("tokenizer.json"),
            talker_config_path: temp.join("config.json"),
            talker_weights_path: temp.join("weights/talker.safetensors"),
            generation_config: Qwen3TtsGenerationConfigSource::Path(
                temp.join("generation_config.json"),
            ),
            codec_config_path: temp.join("configs/codec.json"),
            codec_weights_path: temp.join("weights/codec.safetensors"),
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
            .compile_request(&crate::QwenRequest::Base(crate::BaseRequest::new(
                "hello from the compiler test prompt with many tokens",
            )))
            .unwrap();

        assert!(condition.text_token_ids.len() >= 8);
        assert_eq!(
            condition.controls.codec_prefix_ids,
            vec![2051, 2052, 2053, 2049, 2048]
        );
        assert_eq!(condition.codec_eos_token_id, 2150);
    }

    #[test]
    fn compile_request_accepts_case_insensitive_language_names() {
        let compiler = compiler_fixture();

        let condition = compiler
            .compile_request(&crate::QwenRequest::Base(crate::BaseRequest {
                text: "hello from the compiler test prompt with enough tokens for chinese"
                    .to_string(),
                language: crate::LanguageSelection::Named("Chinese".to_string()),
            }))
            .unwrap();

        assert_eq!(
            condition.controls.codec_prefix_ids,
            vec![2050, 2052, 3001, 2053, 2049, 2048]
        );
    }

    #[test]
    fn compile_request_includes_custom_voice_speaker_id() {
        let compiler = compiler_fixture();

        let condition = compiler
            .compile_request(&crate::QwenRequest::CustomVoice(
                crate::CustomVoiceRequest {
                    text: "ni hao from the compiler test prompt".to_string(),
                    language: crate::LanguageSelection::Named("Chinese".to_string()),
                    speaker: Some("Chelsie".to_string()),
                },
            ))
            .unwrap();

        assert!(condition.text_token_ids.len() >= 8);
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
        write_generation_config(&temp.join("generation_config.json"));
        write_model_config(&temp.join("config.json"));
        write_tokenizer_file(&temp.join("tokenizer.json"));
        Qwen3TtsRequestCompiler::load(&Qwen3TtsPackage {
            package_root: temp.clone(),
            name: "compiler-fixture".to_string(),
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
}
