use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use thiserror::Error;

use crate::audio_codec::{
    LoadedQwen3TtsAudioCodec, Qwen3TtsAudioCodecConfig, decode_codec_tokens,
    load_qwen3_tts_audio_codec,
};
use crate::frontend::{
    CustomVoiceBatch, CustomVoiceGenerationConfig, CustomVoiceRequest, FrontendOutput,
    Qwen3TtsTextTokenizer, build_custom_voice_prefill_batch, load_custom_voice_generation_config,
};
use crate::shared::io::{LoadReport, LoadedQwen3TtsTalker, save_wav};
use crate::talker::{
    CodePredictorGenerateInput, KeyValueCache, Qwen3TtsTalkerConfig, SamplingConfig, StoppingRules,
    TalkerGenerateInput, generate_code_predictor_groups, generate_talker_tokens,
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
pub struct Qwen3TtsSynthesisOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

impl Default for Qwen3TtsSynthesisOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::greedy(),
        }
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
    pub talker_hidden_states: Vec<Tensor<B, 2>>,
    pub talker_prefill_logits: Tensor<B, 3>,
    pub talker_step_logits: Vec<Tensor<B, 3>>,
    pub codec_token_ids: Tensor<B, 3, Int>,
    pub generated_audio_steps: usize,
}

#[derive(Debug)]
pub struct Qwen3TtsSynthesisOutput<B: Backend> {
    pub codec_generation: Qwen3TtsCodecGenerationOutput<B>,
    pub waveform: Tensor<B, 3>,
    pub sample_rate: u32,
}

#[derive(Debug)]
pub struct Qwen3TtsPipeline<B: Backend> {
    talker: LoadedQwen3TtsTalker<B>,
    audio_codec: LoadedQwen3TtsAudioCodec<B>,
    tokenizer: Qwen3TtsTextTokenizer,
    generation_config: CustomVoiceGenerationConfig,
    device: B::Device,
    model_dir: PathBuf,
}

impl<B> Qwen3TtsPipeline<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsPipelineError> {
        let model_dir = model_dir.as_ref().to_path_buf();

        let started = Instant::now();
        let talker = load_qwen3_tts_talker_for_inference::<B>(&model_dir, device)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded talker for pipeline"
        );

        let started = Instant::now();
        let audio_codec = load_qwen3_tts_audio_codec::<B>(&model_dir, device)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded audio codec for pipeline"
        );

        let started = Instant::now();
        let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "loaded text tokenizer for pipeline"
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

    pub fn load_report(&self) -> Qwen3TtsPipelineLoadReport {
        Qwen3TtsPipelineLoadReport {
            talker: self.talker.load_report.clone(),
            audio_codec: self.audio_codec.load_report.clone(),
        }
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

    pub fn generate_codec_tokens(
        &self,
        frontend: FrontendOutput<B>,
        options: &Qwen3TtsSynthesisOptions,
    ) -> Result<Qwen3TtsCodecGenerationOutput<B>, Qwen3TtsPipelineError> {
        let [batch_size, _, _] = frontend.inputs_embeds.dims();
        if batch_size != 1 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "pipeline generation currently supports batch size 1, got {batch_size}"
                ),
            }
            .into());
        }
        if options.max_new_tokens == 0 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: "max_new_tokens must be greater than zero".to_string(),
            }
            .into());
        }

        let cfg = &self.talker.config.talker_config;
        let started = Instant::now();
        let mut talker_cache = (0..cfg.num_hidden_layers)
            .map(|_| KeyValueCache::new(1, cfg.num_key_value_heads, 4096, cfg.head_dim))
            .collect::<Vec<_>>();
        let generated = generate_talker_tokens(
            cfg,
            &self.talker,
            TalkerGenerateInput {
                prefill_inputs_embeds: frontend.inputs_embeds,
                prefill_position_ids: frontend.position_ids,
                prefill_attention_mask: Some(frontend.attention_mask),
                trailing_text_hidden: Some(frontend.trailing_text_hidden),
                tts_pad_embed: Some(frontend.tts_pad_embed),
                sampling: options.sampling.clone(),
                stopping: StoppingRules {
                    max_new_tokens: options.max_new_tokens,
                    eos_token_id: Some(self.generation_config.codec_eos_token_id),
                },
                suppress_token_ids: self.generation_config.suppress_token_ids.clone(),
                collect_step_diagnostics: false,
            },
            &mut talker_cache,
        )?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            generated_shape = ?generated.generated_token_ids.dims(),
            hidden_steps = generated.step_hidden_states.len(),
            "generated talker tokens"
        );

        let generated_token_ids = generated
            .generated_token_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .map_err(|e| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read generated token ids: {e}"),
            })?;
        let generated_audio_steps = generated_audio_steps(
            &generated_token_ids,
            self.generation_config.codec_eos_token_id,
        );
        if generated_audio_steps == 0 {
            return Err(Qwen3TtsInferenceError::InvalidInput {
                message: "talker emitted EOS before any audio codec token".to_string(),
            }
            .into());
        }

        let started = Instant::now();
        let mut codec_steps = Vec::with_capacity(generated_audio_steps);
        for step in 0..generated_audio_steps {
            let base_token = generated
                .generated_token_ids
                .clone()
                .slice([0..1, step..step + 1]);
            let hidden = generated.step_hidden_states[step].clone();
            let mut predictor_cache = (0..cfg.code_predictor_config.num_hidden_layers)
                .map(|_| {
                    KeyValueCache::new(
                        1,
                        cfg.code_predictor_config.num_key_value_heads,
                        cfg.num_code_groups + 1,
                        cfg.code_predictor_config.head_dim,
                    )
                })
                .collect::<Vec<_>>();
            let groups = generate_code_predictor_groups(
                cfg,
                &self.talker,
                CodePredictorGenerateInput {
                    talker_hidden_state: hidden,
                    base_codec_token_id: base_token,
                    sampling: options.sampling.clone(),
                    collect_step_diagnostics: false,
                },
                &mut predictor_cache,
            )?;
            codec_steps.push(groups.codec_ids.reshape([1, cfg.num_code_groups, 1]));
        }
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            generated_audio_steps,
            code_groups = cfg.num_code_groups,
            "generated code predictor groups"
        );

        let codec_token_ids = Tensor::cat(codec_steps, 2);

        Ok(Qwen3TtsCodecGenerationOutput {
            talker_token_ids: generated.generated_token_ids,
            talker_hidden_states: generated.step_hidden_states,
            talker_prefill_logits: generated.prefill_logits,
            talker_step_logits: generated.step_logits,
            codec_token_ids,
            generated_audio_steps,
        })
    }

    pub fn decode_codec_tokens(
        &self,
        codec_token_ids: Tensor<B, 3, Int>,
    ) -> Result<Tensor<B, 3>, Qwen3TtsPipelineError> {
        let started = Instant::now();
        let waveform = decode_codec_tokens::<B>(
            &self.audio_codec,
            codec_token_ids,
            &self.audio_codec.config.decoder_config,
        )?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            waveform_shape = ?waveform.dims(),
            "decoded waveform"
        );
        Ok(waveform)
    }

    pub fn synthesize(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsSynthesisOptions,
    ) -> Result<Qwen3TtsSynthesisOutput<B>, Qwen3TtsPipelineError> {
        let frontend = self.build_frontend(request)?;
        let codec_generation = self.generate_codec_tokens(frontend, options)?;
        let waveform = self.decode_codec_tokens(codec_generation.codec_token_ids.clone())?;

        Ok(Qwen3TtsSynthesisOutput {
            codec_generation,
            waveform,
            sample_rate: self.audio_codec.config.output_sample_rate as u32,
        })
    }

    pub fn synthesize_to_wav(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsSynthesisOptions,
        path: impl AsRef<Path>,
    ) -> Result<Qwen3TtsSynthesisOutput<B>, Qwen3TtsPipelineError> {
        let output = self.synthesize(request, options)?;
        save_wav(&output.waveform, path.as_ref(), output.sample_rate)?;
        Ok(output)
    }
}

fn generated_audio_steps(token_ids: &[i32], eos_token_id: usize) -> usize {
    token_ids
        .iter()
        .position(|id| *id as usize == eos_token_id)
        .unwrap_or(token_ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_options_default_to_greedy_generation() {
        let options = Qwen3TtsSynthesisOptions::default();

        assert_eq!(options.max_new_tokens, 256);
        assert!(!options.sampling.do_sample);
        assert_eq!(options.sampling.temperature, 1.0);
    }

    #[test]
    fn generated_audio_steps_stops_at_first_eos() {
        assert_eq!(generated_audio_steps(&[1, 2, 7, 8], 7), 2);
        assert_eq!(generated_audio_steps(&[1, 2, 3], 9), 3);
    }
}
