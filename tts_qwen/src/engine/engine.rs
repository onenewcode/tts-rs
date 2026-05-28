use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use tokenizers::Tokenizer;
use tts_core::runtime::sampling::SamplingConfig;

use crate::engine::config::EngineConfig;
use crate::error::{QwenTtsError, QwenTtsInferenceError};
use crate::frontend::{
    CustomVoiceGenerationConfig, CustomVoiceRequest, compile_request,
    load_custom_voice_generation_config,
};
use crate::io::tokenizer::load_qwen3_tts_tokenizer;
use crate::model::load::audio_codec::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};
use crate::model::load::talker::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};
use crate::model::variant::QwenTtsVariant;
use crate::profiling::{configure, with_session_context};
use crate::runners::codec::{decode_waveform, waveform_to_pcm};
use crate::runners::talker::{TalkerGenerationOutput, TalkerGenerator};

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
pub struct QwenTtsEngine<B: Backend>
where
    B::Device: Clone,
{
    model_dir: PathBuf,
    talker: LoadedQwen3TtsTalker<B>,
    audio_codec: LoadedQwen3TtsAudioCodec<B>,
    tokenizer: Tokenizer,
    generation_config: CustomVoiceGenerationConfig,
    device: B::Device,
}

impl<B> QwenTtsEngine<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
        variant: QwenTtsVariant,
        config: EngineConfig,
    ) -> Result<Self, QwenTtsError> {
        configure(&config.profiling);
        match variant {
            QwenTtsVariant::Qwen3Tts12Hz06BCustomVoice => {}
        }
        let model_dir = model_dir.as_ref().to_path_buf();
        let talker = load_qwen3_tts_talker_for_inference::<B>(&model_dir, device)?;
        let audio_codec = load_qwen3_tts_audio_codec::<B>(&model_dir, device)?;
        let tokenizer =
            load_qwen3_tts_tokenizer(&model_dir).map_err(QwenTtsInferenceError::from)?;
        let generation_config = load_custom_voice_generation_config(&model_dir)?;
        Ok(Self {
            model_dir,
            talker,
            audio_codec,
            tokenizer,
            generation_config,
            device: device.clone(),
        })
    }

    pub fn start_run(
        &self,
        request: CustomVoiceRequest,
        config: QwenRunConfig,
    ) -> Result<QwenRun<B>, QwenTtsError> {
        let compiled = compile_request(
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

    pub fn step_run(&self, run: &mut QwenRun<B>) -> Result<QwenRunStep, QwenTtsError> {
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

    pub fn snapshot_audio(&self, run: &QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_talker(&run.talker)
    }

    pub fn finish_run(&self, run: QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
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
