use std::collections::BTreeMap;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::error::{Qwen3TtsInferenceError, QwenTtsInferenceError};
use crate::model::config::talker::Qwen3TtsTalkerConfig;
use crate::model::load::talker::LoadedQwen3TtsTalker;
use crate::profiling::record_operator;

#[derive(Debug, Clone)]
pub struct CustomVoiceRequest {
    pub text: String,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

impl CustomVoiceRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            speaker: None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CompiledRequest<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub trailing_text_hidden: Tensor<B, 3>,
    pub tts_pad_embed: Tensor<B, 3>,
}

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
    #[allow(dead_code)]
    Bool(bool),
    Dialect(String),
}

#[derive(Debug, Clone)]
struct CustomVoiceControlIds {
    tts_bos_token_id: i64,
    tts_eos_token_id: i64,
    tts_pad_token_id: i64,
    codec_bos_id: i64,
    codec_pad_id: i64,
    codec_prefix_ids: Vec<i64>,
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

pub(crate) fn compile_request<B: Backend>(
    tokenizer: &Tokenizer,
    model_dir: &Path,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    request: &CustomVoiceRequest,
    device: &B::Device,
) -> Result<CompiledRequest<B>, QwenTtsInferenceError> {
    let prompt = build_custom_voice_prompt(request);
    let text_ids = record_operator("frontend.tokenize", || {
        tokenizer.encode(prompt.as_str(), false)
    })
    .map(|encoding| {
        encoding
            .get_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>()
    })
    .map_err(|source| QwenTtsInferenceError::InvalidInput {
        message: format!("failed to tokenize custom voice prompt: {source}"),
    })?;
    if text_ids.len() < 8 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "custom voice prompt tokenization is too short: {} tokens",
                text_ids.len()
            ),
        });
    }

    let controls = resolve_custom_voice_control_ids(model_dir, request)?;
    let sample = record_operator("frontend.sample_embed", || {
        build_non_streaming_custom_voice_sample(
            talker,
            &text_ids,
            &controls,
            talker_config.hidden_size,
            device,
        )
    });
    let tts_pad_embed = record_operator("frontend.tts_pad_embed", || {
        build_tts_pad_embed(talker, controls.tts_pad_token_id, device)
    });
    let trailing_text_hidden = tts_pad_embed.clone();

    let preferred_dtype = preferred_hidden_dtype::<B>(device);
    let seq_len = sample.dims()[1];
    let inputs_embeds = sample.cast(preferred_dtype);
    let attention_mask =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1; seq_len], [1, seq_len]), device);
    let position_data = (0..3)
        .flat_map(|_| (0..seq_len).map(|pos| pos as i32))
        .collect::<Vec<_>>();
    let position_ids =
        Tensor::<B, 3, Int>::from_data(TensorData::new(position_data, [3, 1, seq_len]), device);

    Ok(CompiledRequest {
        inputs_embeds,
        position_ids,
        attention_mask,
        trailing_text_hidden: trailing_text_hidden.cast(preferred_dtype),
        tts_pad_embed: tts_pad_embed.cast(preferred_dtype),
    })
}

fn resolve_custom_voice_control_ids(
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

fn preferred_hidden_dtype<B: Backend>(device: &B::Device) -> DType {
    if B::supports_dtype(device, DType::BF16) {
        DType::BF16
    } else {
        DType::F32
    }
}

fn build_tts_pad_embed<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    tts_pad_token_id: i64,
    device: &B::Device,
) -> Tensor<B, 3> {
    project_text_ids(talker, &[tts_pad_token_id], device)
}

fn build_non_streaming_custom_voice_sample<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    controls: &CustomVoiceControlIds,
    hidden_size: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let special_embeds = project_text_ids(
        talker,
        &[
            controls.tts_bos_token_id,
            controls.tts_eos_token_id,
            controls.tts_pad_token_id,
        ],
        device,
    );
    let tts_bos_embed = special_embeds.clone().slice([0..1, 0..1, 0..hidden_size]);
    let tts_eos_embed = special_embeds.clone().slice([0..1, 1..2, 0..hidden_size]);
    let tts_pad_embed = special_embeds.slice([0..1, 2..3, 0..hidden_size]);

    let role_embeds = project_text_ids(talker, &text_ids[..3], device);
    let body_embeds = project_text_ids(talker, &text_ids[3..text_ids.len() - 5], device);

    let codec_embeds = embed_codec_ids(talker, &controls.codec_prefix_ids, device);
    let codec_len = controls.codec_prefix_ids.len();
    let codec_prefix_embeds = codec_embeds
        .clone()
        .slice([0..1, 0..codec_len - 1, 0..hidden_size]);
    let prefix_text_embeds = Tensor::cat(
        vec![
            tts_pad_embed
                .clone()
                .repeat_dim(1, codec_len.saturating_sub(2)),
            tts_bos_embed,
        ],
        1,
    );
    let prefix_embeds = prefix_text_embeds + codec_prefix_embeds;

    let body_len = body_embeds.dims()[1];
    let text_with_codec_pad = body_embeds
        + embed_codec_ids(
            talker,
            &std::iter::repeat_n(controls.codec_pad_id, body_len).collect::<Vec<_>>(),
            device,
        );
    let eos_with_codec_pad =
        tts_eos_embed + embed_codec_ids(talker, &[controls.codec_pad_id], device);
    let generation_bos = tts_pad_embed + embed_codec_ids(talker, &[controls.codec_bos_id], device);

    Tensor::cat(
        vec![
            role_embeds,
            prefix_embeds,
            text_with_codec_pad,
            eos_with_codec_pad,
            generation_bos,
        ],
        1,
    )
}

fn project_text_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            [1, ids.len()],
        ),
        device,
    );
    let embeds = talker.model.talker.model.text_embedding.forward(tensor);
    talker.model.talker.text_projection.forward(embeds)
}

fn embed_codec_ids<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    ids: &[i64],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            [1, ids.len()],
        ),
        device,
    );
    talker.model.talker.model.codec_embedding.forward(tensor)
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}
