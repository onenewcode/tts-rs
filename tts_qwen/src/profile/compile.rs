use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};
use tokenizers::Tokenizer;

use crate::error::QwenTtsInferenceError;
use crate::arch::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::arch::engine::components::generator::weights::LoadedQwen3TtsTalker;
use crate::profile::QwenRequest;
use crate::profile::base::BaseRequest;
use crate::profile::base::config::resolve_base_control_ids;
use crate::profile::base::prompt::build_base_prompt;
use crate::profile::custom_voice::config::resolve_custom_voice_control_ids;
use crate::profile::custom_voice::prompt::build_custom_voice_prompt;
use crate::profile::custom_voice::request::CustomVoiceRequest;
use crate::profile::model_config::ProfileControlIds;
use crate::profiling::record_operator;
use crate::releases::{QwenProfile, QwenReleaseManifest};

#[derive(Debug)]
pub(crate) struct CompiledRequest<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub trailing_text_hidden: Tensor<B, 3>,
    pub tts_pad_embed: Tensor<B, 3>,
}

pub(crate) fn compile_request<B: Backend>(
    release: &'static QwenReleaseManifest,
    tokenizer: &Tokenizer,
    model_dir: &Path,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    request: &QwenRequest,
    device: &B::Device,
) -> Result<CompiledRequest<B>, QwenTtsInferenceError> {
    match release.profile {
        QwenProfile::Base => {
            let request = BaseRequest::from(request.clone());
            let prompt = build_base_prompt(&request);
            let controls = resolve_base_control_ids(model_dir, &request, request.source())?;
            compile_qwen3_tts_request(tokenizer, talker_config, talker, prompt, &controls, device)
        }
        QwenProfile::CustomVoice => {
            let request = CustomVoiceRequest::from(request.clone());
            let prompt = build_custom_voice_prompt(&request);
            let controls = resolve_custom_voice_control_ids(model_dir, &request)?;
            compile_qwen3_tts_request(tokenizer, talker_config, talker, prompt, &controls, device)
        }
    }
}

fn compile_qwen3_tts_request<B: Backend>(
    tokenizer: &Tokenizer,
    talker_config: &Qwen3TtsTalkerConfig,
    talker: &LoadedQwen3TtsTalker<B>,
    prompt: String,
    controls: &ProfileControlIds,
    device: &B::Device,
) -> Result<CompiledRequest<B>, QwenTtsInferenceError> {
    let text_ids = record_operator("profile.tokenize", || {
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
        message: format!("failed to tokenize qwen prompt: {source}"),
    })?;
    if text_ids.len() < 8 {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "qwen prompt tokenization is too short: {} tokens",
                text_ids.len()
            ),
        });
    }

    let sample = record_operator("profile.sample_embed", || {
        build_non_streaming_sample(
            talker,
            &text_ids,
            controls,
            talker_config.hidden_size,
            device,
        )
    });
    let tts_pad_embed = record_operator("profile.tts_pad_embed", || {
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

fn build_non_streaming_sample<B: Backend>(
    talker: &LoadedQwen3TtsTalker<B>,
    text_ids: &[i64],
    controls: &ProfileControlIds,
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
