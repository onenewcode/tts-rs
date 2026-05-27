use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;

use crate::engine::config::EngineConfig;
use crate::error::{QwenTtsError, QwenTtsInferenceError};
use crate::io::tokenizer::Qwen3TtsTextTokenizer;
use crate::model::load::audio_codec::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};
use crate::model::load::talker::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};
use crate::profiling::{configure, with_session_context};
use crate::runners::codec::{decode_waveform, waveform_to_pcm};
use crate::runners::frontend::compile_request;
use crate::runners::talker::{TalkerGenerationOutput, TalkerGenerator};
use crate::scheduler::SingleSessionScheduler;
use crate::session::{
    AudioChunk, CustomVoiceGenerationConfig, CustomVoiceRequest, FinishedSession, SessionConfig,
    SessionHandle, StreamEvent, StreamingMode, TtsSession, TtsSessionState,
    load_custom_voice_generation_config,
};

#[derive(Debug, Clone)]
pub enum StepOutcome {
    MadeProgress,
    ProducedEvents(usize),
    Finished,
}

#[derive(Debug, Clone)]
pub struct FinishedInference {
    pub sample_rate: u32,
    pub generated_audio_steps: usize,
    pub talker_token_count: usize,
    pub waveform_pcm: Vec<i16>,
}

#[derive(Debug)]
pub struct QwenTtsEngine<B: Backend>
where
    B::Device: Clone,
{
    config: EngineConfig,
    model_dir: PathBuf,
    talker: LoadedQwen3TtsTalker<B>,
    audio_codec: LoadedQwen3TtsAudioCodec<B>,
    tokenizer: Qwen3TtsTextTokenizer,
    generation_config: CustomVoiceGenerationConfig,
    device: B::Device,
    sessions: Vec<Option<TtsSession<B>>>,
    scheduler: SingleSessionScheduler,
}

impl<B> QwenTtsEngine<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
        config: EngineConfig,
    ) -> Result<Self, QwenTtsError> {
        configure(&config.profiling);
        let model_dir = model_dir.as_ref().to_path_buf();
        let talker = load_qwen3_tts_talker_for_inference::<B>(&model_dir, device)?;
        let audio_codec = load_qwen3_tts_audio_codec::<B>(&model_dir, device)?;
        let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir)
            .map_err(QwenTtsInferenceError::from)?;
        let generation_config = load_custom_voice_generation_config(&model_dir)?;
        Ok(Self {
            config,
            model_dir,
            talker,
            audio_codec,
            tokenizer,
            generation_config,
            device: device.clone(),
            sessions: Vec::new(),
            scheduler: SingleSessionScheduler,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn start_session(
        &mut self,
        request: CustomVoiceRequest,
        config: SessionConfig,
    ) -> Result<SessionHandle, QwenTtsError> {
        if self
            .sessions
            .iter()
            .filter(|session| session.is_some())
            .count()
            >= self.config.max_concurrent_sessions
        {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "max_concurrent_sessions={} exceeded",
                    self.config.max_concurrent_sessions
                ),
            }
            .into());
        }
        let compiled = compile_request(
            &self.tokenizer,
            &self.talker.config.talker_config,
            &self.talker,
            &request,
            &self.device,
        )?;
        let id = self
            .sessions
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.sessions.len());
        let session = TtsSession {
            id,
            state: TtsSessionState::Created,
            config,
            compiled,
            talker: None,
            pending_audio: Default::default(),
            queued_events: Vec::new(),
        };
        if id == self.sessions.len() {
            self.sessions.push(Some(session));
        } else {
            self.sessions[id] = Some(session);
        }
        Ok(SessionHandle(id))
    }

    pub fn step(&mut self, handle: SessionHandle) -> Result<StepOutcome, QwenTtsError> {
        let mut session = self.take_session(handle)?;
        if session.state == TtsSessionState::Finished {
            self.put_session(handle, session);
            return Ok(StepOutcome::Finished);
        }
        if session.talker.is_none() {
            session.talker = Some(TalkerGenerator::start(
                &self.talker.config.talker_config,
                &self.talker,
                &session.compiled,
                session.config.sampling.clone(),
                session.config.max_new_tokens,
                Some(self.generation_config.codec_eos_token_id),
                self.generation_config.suppress_token_ids.clone(),
            )?);
            session.state = TtsSessionState::Prefilled;
        }

        let session_id = session.id;
        let step_idx = session
            .talker
            .as_ref()
            .map(TalkerGenerator::step_idx)
            .unwrap_or_default();
        let step_result = with_session_context(session_id, step_idx, || {
            session
                .talker
                .as_mut()
                .expect("talker session should exist")
                .step(&self.talker)
        })?;
        let outcome = if let Some(step) = step_result {
            session.state = if step.finished {
                TtsSessionState::Draining
            } else {
                TtsSessionState::Generating
            };
            session
                .queued_events
                .push(StreamEvent::CodecChunk { steps: 1 });
            self.maybe_emit_audio(&mut session, step.finished)?;
            if step.finished {
                session.queued_events.push(StreamEvent::Finished);
                session.state = TtsSessionState::Finished;
                StepOutcome::Finished
            } else if session.queued_events.is_empty() {
                StepOutcome::MadeProgress
            } else {
                StepOutcome::ProducedEvents(session.queued_events.len())
            }
        } else {
            self.maybe_emit_audio(&mut session, true)?;
            session.queued_events.push(StreamEvent::Finished);
            session.state = TtsSessionState::Finished;
            StepOutcome::Finished
        };
        self.put_session(handle, session);
        Ok(outcome)
    }

    pub fn drain_events(
        &mut self,
        handle: SessionHandle,
    ) -> Result<Vec<StreamEvent>, QwenTtsError> {
        let mut session = self.take_session(handle)?;
        let events = std::mem::take(&mut session.queued_events);
        self.put_session(handle, session);
        Ok(events)
    }

    pub fn run_to_end(&mut self, handle: SessionHandle) -> Result<FinishedInference, QwenTtsError> {
        loop {
            if matches!(self.step(handle)?, StepOutcome::Finished) {
                break;
            }
            let _ = self.drain_events(handle)?;
        }
        self.finish_session(handle)
    }

    pub fn finish_session(
        &mut self,
        handle: SessionHandle,
    ) -> Result<FinishedInference, QwenTtsError> {
        let mut session = self.take_session(handle)?;
        let finished = self.decode_finished_session(&mut session)?;
        Ok(FinishedInference {
            sample_rate: finished.sample_rate,
            generated_audio_steps: finished.generated_audio_steps,
            talker_token_count: finished.talker_token_count,
            waveform_pcm: finished.waveform_pcm,
        })
    }

    fn maybe_emit_audio(
        &self,
        session: &mut TtsSession<B>,
        finished: bool,
    ) -> Result<(), QwenTtsError> {
        if session.config.streaming == StreamingMode::Full {
            return Ok(());
        }
        let talker = session
            .talker
            .as_ref()
            .expect("talker session should exist");
        let generated_steps = talker.generated_audio_steps();
        let pending_steps = generated_steps.saturating_sub(session.pending_audio.emitted_steps);
        if !self.scheduler.should_emit_audio_chunk(
            pending_steps,
            self.config.codec_chunk_steps,
            finished,
        ) {
            return Ok(());
        }
        let generation = talker.finalize()?;
        let waveform = decode_waveform(&self.audio_codec, generation.codec_token_ids)?;
        let pcm = waveform_to_pcm(&waveform)?;
        let delta = pcm[session.pending_audio.emitted_samples..].to_vec();
        session.pending_audio.emitted_steps = generated_steps;
        session.pending_audio.emitted_samples = pcm.len();
        session
            .queued_events
            .push(StreamEvent::AudioChunk(AudioChunk {
                pcm: delta,
                sample_rate: self.audio_codec.config.output_sample_rate as u32,
                is_final: finished,
            }));
        Ok(())
    }

    fn decode_finished_session(
        &self,
        session: &mut TtsSession<B>,
    ) -> Result<FinishedSession, QwenTtsError> {
        let generation: TalkerGenerationOutput<B> = session
            .talker
            .as_ref()
            .expect("talker session should exist")
            .finalize()?;
        let waveform = decode_waveform(&self.audio_codec, generation.codec_token_ids)?;
        let pcm = waveform_to_pcm(&waveform)?;
        Ok(FinishedSession {
            sample_rate: self.audio_codec.config.output_sample_rate as u32,
            generated_audio_steps: generation.generated_audio_steps,
            talker_token_count: generation.talker_token_ids.dims()[1],
            waveform_pcm: pcm,
        })
    }

    fn take_session(&mut self, handle: SessionHandle) -> Result<TtsSession<B>, QwenTtsError> {
        self.sessions
            .get_mut(handle.0)
            .and_then(Option::take)
            .ok_or_else(|| {
                QwenTtsInferenceError::InvalidInput {
                    message: format!("unknown session handle {}", handle.0),
                }
                .into()
            })
    }

    fn put_session(&mut self, handle: SessionHandle, session: TtsSession<B>) {
        if handle.0 == self.sessions.len() {
            self.sessions.push(Some(session));
        } else {
            self.sessions[handle.0] = Some(session);
        }
    }
}
