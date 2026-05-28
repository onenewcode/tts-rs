use std::path::Path;

use crate::Qwen3TtsInferenceError;
use crate::profile::QwenRequest;
use crate::profile::model_config::{
    GenerationConfig, ProfileControlIds, build_generation_config, load_model_prompt_config,
    normalize_key,
};

use super::request::BaseRequest;

pub(crate) fn load_base_generation_config(
    model_dir: &Path,
) -> Result<GenerationConfig, Qwen3TtsInferenceError> {
    build_generation_config(model_dir)
}

pub(crate) fn resolve_base_control_ids(
    model_dir: &Path,
    request: &BaseRequest,
    source: &QwenRequest,
) -> Result<ProfileControlIds, Qwen3TtsInferenceError> {
    if source
        .speaker
        .as_deref()
        .is_some_and(|speaker| !speaker.trim().is_empty())
    {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "speaker is not supported for qwen base profile".to_string(),
        });
    }

    let config = load_model_prompt_config(model_dir)?;
    let talker = &config.talker_config;
    let language = request
        .language
        .as_deref()
        .map(normalize_key)
        .unwrap_or_else(|| "auto".to_string());

    let codec_prefix_ids = if language == "auto" {
        vec![
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
            talker.codec_pad_id,
            talker.codec_bos_id,
        ]
    } else {
        let language_id = *talker.codec_language_id.get(&language).ok_or_else(|| {
            Qwen3TtsInferenceError::InvalidInput {
                message: format!("unsupported language: {language}"),
            }
        })?;
        vec![
            talker.codec_think_id,
            talker.codec_think_bos_id,
            language_id,
            talker.codec_think_eos_id,
            talker.codec_pad_id,
            talker.codec_bos_id,
        ]
    };

    Ok(ProfileControlIds {
        tts_bos_token_id: config.tts_bos_token_id,
        tts_eos_token_id: config.tts_eos_token_id,
        tts_pad_token_id: config.tts_pad_token_id,
        codec_bos_id: talker.codec_bos_id,
        codec_pad_id: talker.codec_pad_id,
        codec_prefix_ids,
    })
}
