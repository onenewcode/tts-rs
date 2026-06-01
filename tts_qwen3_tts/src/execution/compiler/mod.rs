mod prompt;
pub(crate) mod session_seed;
mod tokenizer;

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    LanguageSelection, Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource,
    Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsPackage, QwenRequest,
};

use self::prompt::CompileProfileConditionInput;

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
        let model_kind = detect_model_kind(package, &control_config);

        let base = match model_kind {
            Qwen3TtsModelKind::Base => Some(Qwen3TtsCompiledProfile {
                generation_config: generation_config.clone(),
                control_config: control_config.clone(),
            }),
            Qwen3TtsModelKind::CustomVoice => None,
        };
        let custom_voice = match model_kind {
            Qwen3TtsModelKind::Base => None,
            Qwen3TtsModelKind::CustomVoice => Some(Qwen3TtsCompiledProfile {
                generation_config,
                control_config,
            }),
        };

        Ok(Self {
            tokenizer: tokenizer::load_qwen3_tts_tokenizer(&package.tokenizer_path).map_err(
                |source| Qwen3TtsLoadError::Tokenizer {
                    path: package.tokenizer_path.clone(),
                    source,
                },
            )?,
            profiles: Qwen3TtsCompiledProfiles { base, custom_voice },
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
                        message: "model does not support base requests".to_string(),
                    }
                })?;
                let (prompt_recipe, prompt, ref_prompt, voice_clone) =
                    resolve_base_prompt_recipe(request)?;
                compile_profile_condition(
                    &self.tokenizer,
                    CompileProfileConditionInput {
                        prompt: &prompt,
                        instruct_prompt: None,
                        voice_clone,
                        ref_prompt: ref_prompt.as_deref(),
                        prompt_recipe,
                        controls: resolve_base_control_ids(
                            &profile.control_config,
                            &request.language,
                        )?,
                        max_new_tokens: profile.generation_config.max_new_tokens,
                        codec_eos_token_id: profile.control_config.codec_eos_token_id as usize,
                        sampling: crate::SamplingConfig {
                            do_sample: profile.generation_config.do_sample,
                            temperature: profile.generation_config.temperature,
                            top_k: Some(profile.generation_config.top_k),
                            top_p: profile.generation_config.top_p,
                            seed: None,
                            repetition_penalty: profile.generation_config.repetition_penalty,
                        },
                    },
                )
            }
            QwenRequest::CustomVoice(request) => {
                let profile = self.profiles.custom_voice.as_ref().ok_or_else(|| {
                    Qwen3TtsInferenceError::InvalidInput {
                        message: "model does not support custom-voice requests".to_string(),
                    }
                })?;
                let (prompt_recipe, prompt, instruct_prompt) =
                    resolve_custom_voice_prompt_recipe(request)?;
                compile_profile_condition(
                    &self.tokenizer,
                    CompileProfileConditionInput {
                        prompt: &prompt,
                        instruct_prompt: instruct_prompt.as_deref(),
                        voice_clone: None,
                        ref_prompt: None,
                        prompt_recipe,
                        controls: resolve_custom_voice_control_ids(
                            &profile.control_config,
                            &request.language,
                            request.speaker.as_deref(),
                        )?,
                        max_new_tokens: profile.generation_config.max_new_tokens,
                        codec_eos_token_id: profile.control_config.codec_eos_token_id as usize,
                        sampling: crate::SamplingConfig {
                            do_sample: profile.generation_config.do_sample,
                            temperature: profile.generation_config.temperature,
                            top_k: Some(profile.generation_config.top_k),
                            top_p: profile.generation_config.top_p,
                            seed: None,
                            repetition_penalty: profile.generation_config.repetition_penalty,
                        },
                    },
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen3TtsModelKind {
    Base,
    CustomVoice,
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

fn detect_model_kind(
    package: &Qwen3TtsPackage,
    control_config: &Qwen3TtsControlConfig,
) -> Qwen3TtsModelKind {
    let package_name = normalize_key(&package.name);
    if package_name.contains("customvoice") || package_name.contains("custom_voice") {
        return Qwen3TtsModelKind::CustomVoice;
    }
    if package_name.contains("base") {
        return Qwen3TtsModelKind::Base;
    }
    if control_config.spk_id.is_empty() {
        Qwen3TtsModelKind::Base
    } else {
        Qwen3TtsModelKind::CustomVoice
    }
}

pub(crate) use self::prompt::{
    CompiledVoiceCloneCondition, ProfileControlIds, Qwen3TtsPromptRecipe, SemanticRequestCondition,
};
use self::prompt::{
    compile_profile_condition, resolve_base_prompt_recipe, resolve_custom_voice_prompt_recipe,
};

fn resolve_base_control_ids(
    config: &Qwen3TtsControlConfig,
    language: &LanguageSelection,
) -> Result<ProfileControlIds, Qwen3TtsInferenceError> {
    let mut codec_prefix_ids = match language_key(language)? {
        None => vec![
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ],
        Some(language) => vec![
            config.codec_think_id,
            config.codec_think_bos_id,
            lookup_language_id(config, &language)?,
            config.codec_think_eos_id,
        ],
    };
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
mod tests;
