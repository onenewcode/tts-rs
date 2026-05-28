use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Qwen3TtsInferenceError;

#[derive(Debug, Deserialize)]
pub(crate) struct ModelPromptConfig {
    pub(crate) tts_bos_token_id: i64,
    pub(crate) tts_eos_token_id: i64,
    pub(crate) tts_pad_token_id: i64,
    pub(crate) talker_config: TalkerPromptConfig,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TalkerPromptConfig {
    pub(crate) vocab_size: usize,
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DialectFlag {
    #[allow(dead_code)]
    Bool(bool),
    Dialect(String),
}

#[derive(Debug, Clone)]
pub(crate) struct ProfileControlIds {
    pub(crate) tts_bos_token_id: i64,
    pub(crate) tts_eos_token_id: i64,
    pub(crate) tts_pad_token_id: i64,
    pub(crate) codec_bos_id: i64,
    pub(crate) codec_pad_id: i64,
    pub(crate) codec_prefix_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub codec_eos_token_id: usize,
    pub suppress_token_ids: Vec<usize>,
}

pub(crate) fn load_model_prompt_config(
    model_dir: &Path,
) -> Result<ModelPromptConfig, Qwen3TtsInferenceError> {
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

pub(crate) fn build_generation_config(
    model_dir: &Path,
) -> Result<GenerationConfig, Qwen3TtsInferenceError> {
    let config = load_model_prompt_config(model_dir)?;
    let eos = config.talker_config.codec_eos_token_id as usize;
    let suppress_token_ids = (config.talker_config.vocab_size.saturating_sub(1024)
        ..config.talker_config.vocab_size)
        .filter(|id| *id != eos)
        .collect();
    Ok(GenerationConfig {
        codec_eos_token_id: eos,
        suppress_token_ids,
    })
}

pub(crate) fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}
