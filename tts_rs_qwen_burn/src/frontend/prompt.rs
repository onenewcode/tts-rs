use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Qwen3TtsInferenceError;

use super::types::CustomVoiceRequest;

#[derive(Debug, Deserialize)]
struct ModelPromptConfig {
    tts_bos_token_id: i64,
    tts_eos_token_id: i64,
    tts_pad_token_id: i64,
    talker_config: TalkerPromptConfig,
}

#[derive(Debug, Deserialize)]
struct TalkerPromptConfig {
    vocab_size: usize,
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
    #[serde(default)]
    spk_is_dialect: BTreeMap<String, DialectFlag>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DialectFlag {
    Bool(bool),
    Dialect(String),
}

#[derive(Debug, Clone)]
pub(crate) struct CustomVoiceControlIds {
    pub tts_bos_token_id: i64,
    pub tts_eos_token_id: i64,
    pub tts_pad_token_id: i64,
    pub codec_bos_id: i64,
    pub codec_pad_id: i64,
    pub codec_prefix_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct CustomVoiceGenerationConfig {
    pub codec_eos_token_id: usize,
    pub suppress_token_ids: Vec<usize>,
}

pub fn build_custom_voice_prompt(request: &CustomVoiceRequest) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        request.text
    )
}

pub fn load_custom_voice_generation_config(
    model_dir: &Path,
) -> Result<CustomVoiceGenerationConfig, Qwen3TtsInferenceError> {
    let config = load_model_prompt_config(model_dir)?;
    let eos = config.talker_config.codec_eos_token_id as usize;
    let suppress_token_ids = (config.talker_config.vocab_size.saturating_sub(1024)
        ..config.talker_config.vocab_size)
        .filter(|id| *id != eos)
        .collect();
    Ok(CustomVoiceGenerationConfig {
        codec_eos_token_id: eos,
        suppress_token_ids,
    })
}

pub(crate) fn resolve_custom_voice_control_ids(
    model_dir: &Path,
    request: &CustomVoiceRequest,
) -> Result<CustomVoiceControlIds, Qwen3TtsInferenceError> {
    let config = load_model_prompt_config(model_dir)?;
    let talker = &config.talker_config;

    let speaker = request.speaker.as_deref().map(normalize_key);
    let language = request
        .language
        .as_deref()
        .map(normalize_key)
        .unwrap_or_else(|| "auto".to_string());
    let mut language_id = if language == "auto" {
        None
    } else {
        Some(
            *talker
                .codec_language_id
                .get(&language)
                .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                    message: format!("unsupported language: {language}"),
                })?,
        )
    };

    if matches!(language.as_str(), "chinese" | "auto") {
        if let Some(speaker_name) = &speaker {
            if let Some(dialect_flag) = talker.spk_is_dialect.get(speaker_name) {
                match dialect_flag {
                    DialectFlag::Dialect(dialect) => {
                        language_id =
                            Some(*talker.codec_language_id.get(dialect).ok_or_else(|| {
                                Qwen3TtsInferenceError::InvalidInput {
                                    message: format!(
                                        "speaker {speaker_name} references unsupported dialect {dialect}"
                                    ),
                                }
                            })?);
                    }
                    DialectFlag::Bool(_is_dialect) => {}
                }
            }
        }
    }

    let mut codec_prefix_ids = if let Some(language_id) = language_id {
        vec![
            talker.codec_think_id,
            talker.codec_think_bos_id,
            language_id,
            talker.codec_think_eos_id,
        ]
    } else {
        vec![
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
        ]
    };

    if let Some(speaker_name) = &speaker {
        if !speaker_name.is_empty() {
            codec_prefix_ids.push(*talker.spk_id.get(speaker_name).ok_or_else(|| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: format!("unsupported speaker: {speaker_name}"),
                }
            })?);
        }
    }
    codec_prefix_ids.extend([talker.codec_pad_id, talker.codec_bos_id]);

    Ok(CustomVoiceControlIds {
        tts_bos_token_id: config.tts_bos_token_id,
        tts_eos_token_id: config.tts_eos_token_id,
        tts_pad_token_id: config.tts_pad_token_id,
        codec_bos_id: talker.codec_bos_id,
        codec_pad_id: talker.codec_pad_id,
        codec_prefix_ids,
    })
}

fn load_model_prompt_config(model_dir: &Path) -> Result<ModelPromptConfig, Qwen3TtsInferenceError> {
    let config_path = model_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path).map_err(|source| {
        Qwen3TtsInferenceError::InvalidInput {
            message: format!("failed to read {}: {source}", config_path.display()),
        }
    })?;
    let config: ModelPromptConfig = serde_json::from_str(&config_text).map_err(|source| {
        Qwen3TtsInferenceError::InvalidInput {
            message: format!("failed to parse {}: {source}", config_path.display()),
        }
    })?;
    Ok(config)
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}
