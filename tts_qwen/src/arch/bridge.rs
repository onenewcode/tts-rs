use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use tokenizers::Tokenizer;
use tts_core::runtime::sampling::SamplingConfig;

use crate::error::{QwenTtsError, QwenTtsInferenceError};
use crate::io::tokenizer::load_qwen3_tts_tokenizer;
use crate::profiling::{configure, with_session_context};
use crate::profile::QwenRequest;
use crate::profile::model_config::GenerationConfig;
use crate::profile::compile::{CompiledRequest, compile_request};
use crate::releases::QwenReleaseManifest;
use crate::runtime::types::EngineConfig;

use super::audio::{decode_waveform, waveform_to_pcm};
use super::load::audio_codec::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};
use super::load::talker::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};
use super::runner::{TalkerGenerationOutput, TalkerGenerator};

#[derive(Debug, Clone)]
pub(crate) struct QwenEngineBridge;

#[derive(Debug, Clone)]
pub(crate) struct QwenRunConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenRunStep {
    pub generated_steps: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FinishedInference {
    pub sample_rate: u32,
    pub waveform_pcm: Vec<i16>,
}

#[derive(Debug)]
pub(crate) struct QwenRun<B: Backend> {
    id: usize,
    talker: TalkerGenerator<B>,
}

#[derive(Debug)]
pub(crate) struct QwenEngine<B: Backend>
where
    B::Device: Clone,
{
    model_dir: PathBuf,
    release: &'static QwenReleaseManifest,
    talker: LoadedQwen3TtsTalker<B>,
    audio_codec: LoadedQwen3TtsAudioCodec<B>,
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: B::Device,
}

impl QwenEngineBridge {
    pub(crate) fn load_engine<B: Backend>(
        model_dir: impl AsRef<Path>,
        release: &'static QwenReleaseManifest,
        device: &B::Device,
        config: EngineConfig,
    ) -> Result<QwenEngine<B>, QwenTtsError>
    where
        B::Device: Clone,
    {
        configure(&config.profiling);
        let model_dir = model_dir.as_ref().to_path_buf();
        let talker = load_qwen3_tts_talker_for_inference::<B>(&model_dir, device)?;
        let audio_codec = load_qwen3_tts_audio_codec::<B>(&model_dir, device)?;
        let tokenizer =
            load_qwen3_tts_tokenizer(&model_dir).map_err(QwenTtsInferenceError::from)?;
        let generation_config = (release.architecture.load_generation_config)(&model_dir, release.profile)?;
        Ok(QwenEngine {
            model_dir,
            release,
            talker,
            audio_codec,
            tokenizer,
            generation_config,
            device: device.clone(),
        })
    }
}

impl<B> QwenEngine<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn start_run(
        &self,
        request: QwenRequest,
        config: QwenRunConfig,
    ) -> Result<QwenRun<B>, QwenTtsError> {
        let compiled: CompiledRequest<B> = compile_request(
            self.release,
            &self.tokenizer,
            &self.model_dir,
            &self.talker.config.talker_config,
            &self.talker,
            &request,
            &self.device,
        )?;
        let talker = TalkerGenerator::start(
            &self.talker.config.talker_config,
            &self.talker,
            &compiled,
            config.sampling,
            config.max_new_tokens,
            Some(self.generation_config.codec_eos_token_id),
            self.generation_config.suppress_token_ids.clone(),
        )?;
        Ok(QwenRun { id: 0, talker })
    }

    pub(crate) fn step_run(&self, run: &mut QwenRun<B>) -> Result<QwenRunStep, QwenTtsError> {
        let step_idx = run.talker.step_idx();
        let step_result = with_session_context(run.id, step_idx, || run.talker.step(&self.talker))?;
        match step_result {
            Some(step) => Ok(QwenRunStep {
                generated_steps: 1,
                finished: step.finished,
            }),
            None => Ok(QwenRunStep {
                generated_steps: 0,
                finished: true,
            }),
        }
    }

    pub(crate) fn snapshot_audio(&self, run: &QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_talker(&run.talker)
    }

    pub(crate) fn finish_run(&self, run: QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_talker(&run.talker)
    }

    fn decode_finished_talker(
        &self,
        talker: &TalkerGenerator<B>,
    ) -> Result<FinishedInference, QwenTtsError> {
        let generation: TalkerGenerationOutput<B> = talker.finalize()?;
        let waveform = decode_waveform(&self.audio_codec, generation.codec_token_ids)?;
        let pcm = waveform_to_pcm(&waveform)?;
        Ok(FinishedInference {
            sample_rate: self.audio_codec.config.output_sample_rate as u32,
            waveform_pcm: pcm,
        })
    }
}
