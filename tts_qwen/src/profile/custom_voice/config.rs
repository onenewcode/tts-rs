use std::path::Path;

use crate::Qwen3TtsInferenceError;
use crate::profile::custom_voice::request::CustomVoiceRequest;
use crate::profile::model_config::{
    DialectFlag, GenerationConfig, ProfileControlIds, build_generation_config,
    load_model_prompt_config, normalize_key,
};

pub type CustomVoiceGenerationConfig = GenerationConfig;

pub fn load_custom_voice_generation_config(
    model_dir: &Path,
) -> Result<CustomVoiceGenerationConfig, Qwen3TtsInferenceError> {
    build_generation_config(model_dir)
}

pub(crate) fn resolve_custom_voice_control_ids(
    model_dir: &Path,
    request: &CustomVoiceRequest,
) -> Result<ProfileControlIds, Qwen3TtsInferenceError> {
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
        Some(*talker.codec_language_id.get(&language).ok_or_else(|| {
            Qwen3TtsInferenceError::InvalidInput {
                message: format!("unsupported language: {language}"),
            }
        })?)
    };

    let dialect_speaker = matches!(language.as_str(), "chinese" | "auto")
        .then(|| {
            speaker.as_ref().and_then(|speaker_name| {
                talker
                    .spk_is_dialect
                    .get(speaker_name)
                    .map(|dialect_flag| (speaker_name, dialect_flag))
            })
        })
        .flatten();
    if let Some((speaker_name, DialectFlag::Dialect(dialect))) = dialect_speaker {
        language_id = Some(*talker.codec_language_id.get(dialect).ok_or_else(|| {
            Qwen3TtsInferenceError::InvalidInput {
                message: format!("speaker {speaker_name} references unsupported dialect {dialect}"),
            }
        })?);
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

    if let Some(speaker_name) = speaker
        .as_ref()
        .filter(|speaker_name| !speaker_name.is_empty())
    {
        codec_prefix_ids.push(*talker.spk_id.get(speaker_name).ok_or_else(|| {
            Qwen3TtsInferenceError::InvalidInput {
                message: format!("unsupported speaker: {speaker_name}"),
            }
        })?);
    }
    codec_prefix_ids.extend([talker.codec_pad_id, talker.codec_bos_id]);

    Ok(ProfileControlIds {
        tts_bos_token_id: config.tts_bos_token_id,
        tts_eos_token_id: config.tts_eos_token_id,
        tts_pad_token_id: config.tts_pad_token_id,
        codec_bos_id: talker.codec_bos_id,
        codec_pad_id: talker.codec_pad_id,
        codec_prefix_ids,
    })
}
