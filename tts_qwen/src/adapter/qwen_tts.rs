use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use thiserror::Error;

use crate::audio_codec::{
    LoadedQwen3TtsAudioCodec, Qwen3TtsAudioCodecConfig, infer as infer_audio_codec,
    load_qwen3_tts_audio_codec,
};
use crate::core::{LocalInferenceOptions, LocalInferenceProfile, LocalModelAdapter};
use crate::frontend::{
    CustomVoiceBatch, CustomVoiceGenerationConfig, CustomVoiceRequest, FrontendOutput,
    Qwen3TtsTextTokenizer, build_custom_voice_prefill_batch, load_custom_voice_generation_config,
};
use crate::shared::io::{LoadReport, LoadedQwen3TtsTalker, save_wav};
use crate::shared::runtime::cache::KeyValueCache;
use crate::talker::{
    Qwen3TtsTalkerConfig, TalkerInferInput, TalkerInferOutput, infer as infer_talker,
    load_qwen3_tts_talker_for_inference,
};
use crate::{Qwen3TtsInferenceError, Qwen3TtsLoadError};

#[derive(Debug, Error)]
pub enum Qwen3TtsPipelineError {
    #[error(transparent)]
    Load(#[from] Qwen3TtsLoadError),
    #[error(transparent)]
    Inference(#[from] Qwen3TtsInferenceError),
}

impl From<tokenizers::Error> for Qwen3TtsPipelineError {
    fn from(source: tokenizers::Error) -> Self {
        Self::Inference(Qwen3TtsInferenceError::from(source))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsPipelineLoadReport {
    pub talker: LoadReport,
    pub audio_codec: LoadReport,
}

#[derive(Debug)]
pub struct Qwen3TtsCodecGenerationOutput<B: Backend> {
    pub talker_token_ids: Tensor<B, 2, Int>,
    pub codec_token_ids: Tensor<B, 3, Int>,
    pub generated_audio_steps: usize,
}

#[derive(Debug)]
pub struct Qwen3TtsInferOutput<B: Backend> {
    pub codec_generation: Qwen3TtsCodecGenerationOutput<B>,
    pub waveform: Tensor<B, 3>,
    pub sample_rate: u32,
}

#[derive(Debug)]
pub struct QwenTtsAdapter<B: Backend> {
    talker: LoadedQwen3TtsTalker<B>,
    audio_codec: LoadedQwen3TtsAudioCodec<B>,
    tokenizer: Qwen3TtsTextTokenizer,
    generation_config: CustomVoiceGenerationConfig,
    device: B::Device,
    model_dir: PathBuf,
}

impl<B> QwenTtsAdapter<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn text_tokenizer(&self) -> &Qwen3TtsTextTokenizer {
        &self.tokenizer
    }

    pub fn talker_config(&self) -> &Qwen3TtsTalkerConfig {
        &self.talker.config.talker_config
    }

    pub fn audio_codec_config(&self) -> &Qwen3TtsAudioCodecConfig {
        &self.audio_codec.config
    }

    pub fn generation_config(&self) -> &CustomVoiceGenerationConfig {
        &self.generation_config
    }

    pub fn build_frontend(
        &self,
        request: &CustomVoiceRequest,
    ) -> Result<FrontendOutput<B>, Qwen3TtsPipelineError> {
        self.build_frontend_batch(&CustomVoiceBatch::single(request.clone()))
    }

    pub fn build_frontend_batch(
        &self,
        batch: &CustomVoiceBatch,
    ) -> Result<FrontendOutput<B>, Qwen3TtsPipelineError> {
        build_custom_voice_prefill_batch(
            &self.tokenizer,
            &self.talker.config.talker_config,
            &self.talker,
            batch,
            &self.device,
        )
        .map_err(Into::into)
    }

    fn infer_codec_tokens(
        &self,
        frontend: FrontendOutput<B>,
        options: &LocalInferenceOptions,
    ) -> Result<Qwen3TtsCodecGenerationOutput<B>, Qwen3TtsPipelineError> {
        let [batch_size, _, _] = frontend.inputs_embeds.dims();
        if batch_size != 1 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "pipeline inference currently supports batch size 1, got {batch_size}"
                ),
            }
            .into());
        }

        let cfg = &self.talker.config.talker_config;
        let mut talker_cache = (0..cfg.num_hidden_layers)
            .map(|_| KeyValueCache::new(1, cfg.num_key_value_heads, 4096, cfg.head_dim))
            .collect::<Vec<_>>();
        let output: TalkerInferOutput<B> = infer_talker(
            cfg,
            &self.talker,
            TalkerInferInput {
                prefill_inputs_embeds: frontend.inputs_embeds,
                prefill_position_ids: frontend.position_ids,
                prefill_attention_mask: Some(frontend.attention_mask),
                trailing_text_hidden: Some(frontend.trailing_text_hidden),
                tts_pad_embed: Some(frontend.tts_pad_embed),
                sampling: options.sampling.clone(),
                max_new_tokens: options.max_new_tokens,
                eos_token_id: Some(self.generation_config.codec_eos_token_id),
                suppress_token_ids: self.generation_config.suppress_token_ids.clone(),
            },
            &mut talker_cache,
        )?;
        tracing::info!(
            talker_token_shape = ?output.talker_token_ids.dims(),
            codec_token_shape = ?output.codec_token_ids.dims(),
            generated_audio_steps = output.generated_audio_steps,
            "generated codec tokens"
        );

        Ok(Qwen3TtsCodecGenerationOutput {
            talker_token_ids: output.talker_token_ids,
            codec_token_ids: output.codec_token_ids,
            generated_audio_steps: output.generated_audio_steps,
        })
    }

    fn infer_waveform(
        &self,
        codec_token_ids: Tensor<B, 3, Int>,
    ) -> Result<Tensor<B, 3>, Qwen3TtsPipelineError> {
        let waveform = infer_audio_codec::<B>(
            &self.audio_codec,
            codec_token_ids,
            &self.audio_codec.config.decoder_config,
        )?;
        tracing::info!(waveform_shape = ?waveform.dims(), "decoded waveform");
        Ok(waveform)
    }
}

impl<B> LocalModelAdapter<B> for QwenTtsAdapter<B>
where
    B: Backend,
    B::Device: Clone,
{
    type Request = CustomVoiceRequest;
    type Output = Qwen3TtsInferOutput<B>;
    type Error = Qwen3TtsPipelineError;
    type LoadReport = Qwen3TtsPipelineLoadReport;

    fn load(model_dir: &Path, device: &B::Device) -> Result<Self, Self::Error> {
        let model_dir = model_dir.to_path_buf();

        let started = Instant::now();
        let talker = load_qwen3_tts_talker_for_inference::<B>(&model_dir, device)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded talker for qwen adapter"
        );

        let started = Instant::now();
        let audio_codec = load_qwen3_tts_audio_codec::<B>(&model_dir, device)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded audio codec for qwen adapter"
        );

        let started = Instant::now();
        let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded text tokenizer for qwen adapter"
        );

        let generation_config = load_custom_voice_generation_config(&model_dir)?;
        tracing::debug!(
            codec_eos_token_id = generation_config.codec_eos_token_id,
            suppress_token_count = generation_config.suppress_token_ids.len(),
            "loaded custom voice generation config"
        );

        Ok(Self {
            talker,
            audio_codec,
            tokenizer,
            generation_config,
            device: device.clone(),
            model_dir,
        })
    }

    fn load_report(&self) -> Self::LoadReport {
        Qwen3TtsPipelineLoadReport {
            talker: self.talker.load_report.clone(),
            audio_codec: self.audio_codec.load_report.clone(),
        }
    }

    fn infer(
        &self,
        request: &Self::Request,
        options: &LocalInferenceOptions,
        profile: &mut LocalInferenceProfile,
    ) -> Result<Self::Output, Self::Error> {
        tracing::info!(
            text_chars = request.text.chars().count(),
            has_language = request.language.is_some(),
            has_speaker = request.speaker.is_some(),
            max_new_tokens = options.max_new_tokens,
            do_sample = options.sampling.do_sample,
            temperature = options.sampling.temperature,
            top_k = ?options.sampling.top_k,
            top_p = ?options.sampling.top_p,
            "starting qwen adapter inference"
        );
        let frontend = profile.record_result("frontend_build", || self.build_frontend(request))?;
        let codec_generation = profile.record_result("talker_generation", || {
            self.infer_codec_tokens(frontend, options)
        })?;
        let waveform = profile.record_result("audio_decode", || {
            self.infer_waveform(codec_generation.codec_token_ids.clone())
        })?;

        tracing::info!(
            generated_audio_steps = codec_generation.generated_audio_steps,
            waveform_shape = ?waveform.dims(),
            sample_rate = self.audio_codec.config.output_sample_rate as u32,
            "finished qwen adapter inference"
        );
        Ok(Qwen3TtsInferOutput {
            codec_generation,
            waveform,
            sample_rate: self.audio_codec.config.output_sample_rate as u32,
        })
    }

    fn write_output(&self, output: &Self::Output, path: &Path) -> Result<(), Self::Error> {
        save_wav(&output.waveform, path, output.sample_rate)?;
        Ok(())
    }
}
