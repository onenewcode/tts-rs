pub(crate) mod session_seed;

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    BaseVoiceCloneConditioning, CustomVoiceRequest, LanguageSelection,
    Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource, Qwen3TtsInferenceError,
    Qwen3TtsLoadError, Qwen3TtsPackage, Qwen3TtsVoiceClonePrompt, Qwen3TtsVoiceClonePromptMode,
    QwenRequest,
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
            tokenizer: crate::io::tokenizer::load_qwen3_tts_tokenizer(&package.tokenizer_path)
                .map_err(|source| Qwen3TtsLoadError::Tokenizer {
                    path: package.tokenizer_path.clone(),
                    source,
                })?,
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
                    &prompt,
                    None,
                    voice_clone,
                    ref_prompt.as_deref(),
                    prompt_recipe,
                    resolve_base_control_ids(&profile.control_config, &request.language)?,
                    profile.generation_config.max_new_tokens,
                    profile.control_config.codec_eos_token_id as usize,
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
                    &prompt,
                    instruct_prompt.as_deref(),
                    None,
                    None,
                    prompt_recipe,
                    resolve_custom_voice_control_ids(
                        &profile.control_config,
                        &request.language,
                        request.speaker.as_deref(),
                    )?,
                    profile.generation_config.max_new_tokens,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3TtsPromptRecipe {
    BasePlain,
    BaseVoiceCloneIcl,
    BaseVoiceCloneXVectorOnly,
    CustomVoicePlain,
    CustomVoiceInstructed,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SemanticRequestCondition {
    pub(crate) text_token_ids: Vec<i64>,
    pub(crate) instruct_token_ids: Option<Vec<i64>>,
    pub(crate) voice_clone: Option<CompiledVoiceCloneCondition>,
    pub(crate) controls: ProfileControlIds,
    pub(crate) max_new_tokens: usize,
    pub(crate) codec_eos_token_id: usize,
    pub(crate) prompt_recipe: Qwen3TtsPromptRecipe,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledVoiceCloneCondition {
    pub(crate) speaker_embedding: Vec<f32>,
    pub(crate) ref_codec_token_ids: Option<Vec<Vec<i64>>>,
    pub(crate) ref_text_token_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen3TtsModelKind {
    Base,
    CustomVoice,
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

fn build_assistant_prompt(text: &str) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        text
    )
}

fn build_ref_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{}<|im_end|>\n", text)
}

fn build_instruct_prompt(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text)
}

fn resolve_base_prompt_recipe(
    request: &crate::BaseRequest,
) -> Result<
    (
        Qwen3TtsPromptRecipe,
        String,
        Option<String>,
        Option<CompiledVoiceCloneCondition>,
    ),
    Qwen3TtsInferenceError,
> {
    match request.voice_clone.as_ref() {
        None => Ok((
            Qwen3TtsPromptRecipe::BasePlain,
            build_assistant_prompt(&request.text),
            None,
            None,
        )),
        Some(BaseVoiceCloneConditioning::ReferenceAudio(_)) => {
            Err(Qwen3TtsInferenceError::InvalidInput {
                message: "reference-audio voice clone inputs must be prepared before compilation"
                    .to_string(),
            })
        }
        Some(BaseVoiceCloneConditioning::Prompt(prompt)) => {
            validate_voice_clone_prompt(prompt)?;
            match prompt.mode {
                Qwen3TtsVoiceClonePromptMode::Icl => Ok((
                    Qwen3TtsPromptRecipe::BaseVoiceCloneIcl,
                    build_assistant_prompt(&request.text),
                    Some(build_ref_prompt(
                        prompt.transcript.as_deref().unwrap_or_default(),
                    )),
                    Some(CompiledVoiceCloneCondition {
                        speaker_embedding: prompt.speaker_embedding.clone(),
                        ref_codec_token_ids: prompt.ref_codec_token_ids.clone(),
                        ref_text_token_ids: None,
                    }),
                )),
                Qwen3TtsVoiceClonePromptMode::XVectorOnly => Ok((
                    Qwen3TtsPromptRecipe::BaseVoiceCloneXVectorOnly,
                    build_assistant_prompt(&request.text),
                    None,
                    Some(CompiledVoiceCloneCondition {
                        speaker_embedding: prompt.speaker_embedding.clone(),
                        ref_codec_token_ids: None,
                        ref_text_token_ids: None,
                    }),
                )),
            }
        }
    }
}

fn resolve_custom_voice_prompt_recipe(
    request: &CustomVoiceRequest,
) -> Result<(Qwen3TtsPromptRecipe, String, Option<String>), Qwen3TtsInferenceError> {
    validate_custom_voice_request(request)?;
    let instruct = request
        .instruct
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match instruct {
        Some(instruct) => Ok((
            Qwen3TtsPromptRecipe::CustomVoiceInstructed,
            build_assistant_prompt(&request.text),
            Some(build_instruct_prompt(instruct)),
        )),
        None => Ok((
            Qwen3TtsPromptRecipe::CustomVoicePlain,
            build_assistant_prompt(&request.text),
            None,
        )),
    }
}

fn validate_custom_voice_request(
    request: &CustomVoiceRequest,
) -> Result<(), Qwen3TtsInferenceError> {
    let speaker = request
        .speaker
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if speaker.is_none() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "custom-voice requests require a non-empty speaker".to_string(),
        });
    }
    Ok(())
}

fn validate_voice_clone_prompt(
    prompt: &Qwen3TtsVoiceClonePrompt,
) -> Result<(), Qwen3TtsInferenceError> {
    if prompt.speaker_embedding.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "voice clone prompt must contain a non-empty speaker embedding".to_string(),
        });
    }
    if matches!(prompt.mode, Qwen3TtsVoiceClonePromptMode::Icl)
        && (prompt
            .transcript
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
            || prompt
                .ref_codec_token_ids
                .as_ref()
                .map(|codes| codes.is_empty())
                .unwrap_or(true))
    {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message:
                "voice clone prompt in ICL mode requires both non-empty transcript and ref codec frames"
                    .to_string(),
        });
    }
    Ok(())
}

fn compile_profile_condition(
    tokenizer: &Tokenizer,
    prompt: &str,
    instruct_prompt: Option<&str>,
    mut voice_clone: Option<CompiledVoiceCloneCondition>,
    ref_prompt: Option<&str>,
    prompt_recipe: Qwen3TtsPromptRecipe,
    controls: ProfileControlIds,
    max_new_tokens: usize,
    codec_eos_token_id: usize,
) -> Result<SemanticRequestCondition, Qwen3TtsInferenceError> {
    let text_token_ids = tokenize_prompt(tokenizer, prompt)?;
    if text_token_ids.len() < 8 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "qwen prompt tokenization is too short: {} tokens",
                text_token_ids.len()
            ),
        });
    }
    if let Some(voice_clone) = voice_clone.as_mut() {
        if matches!(prompt_recipe, Qwen3TtsPromptRecipe::BaseVoiceCloneIcl) {
            let ref_prompt = ref_prompt.ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                message: "voice clone ICL recipe requires a tokenizable ref prompt".to_string(),
            })?;
            voice_clone.ref_text_token_ids = Some(tokenize_prompt(tokenizer, ref_prompt)?);
        }
    }

    Ok(SemanticRequestCondition {
        text_token_ids,
        instruct_token_ids: instruct_prompt
            .map(|prompt| tokenize_prompt(tokenizer, prompt))
            .transpose()?,
        voice_clone,
        controls,
        max_new_tokens,
        codec_eos_token_id,
        prompt_recipe,
    })
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
) -> Result<Vec<i64>, Qwen3TtsInferenceError> {
    Ok(tokenizer
        .encode(prompt, false)
        .map_err(|source| Qwen3TtsInferenceError::Tokenizer { source })?
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>())
}

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
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

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
                text: "hello from the compiler test prompt with enough tokens for chinese"
                    .to_string(),
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
        assert!(
            condition
                .voice_clone
                .as_ref()
                .unwrap()
                .ref_text_token_ids
                .as_ref()
                .is_some()
        );
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
        assert!(
            base_error
                .to_string()
                .contains("does not support custom-voice")
        );

        let custom_voice_error = custom_voice_compiler
            .compile_request(&crate::QwenRequest::Base(crate::BaseRequest::new("hello")))
            .unwrap_err();
        assert!(
            custom_voice_error
                .to_string()
                .contains("does not support base")
        );
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
}
