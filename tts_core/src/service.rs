use std::sync::Arc;

use crate::scheduler::should_emit_audio_chunk;
use crate::{
    ModelRegistry, ModelStep, SynthesisEvent, SynthesisOptions, SynthesisRequest, SynthesisResult,
    TtsCoreError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeSessionState {
    Running,
    Finished,
}

struct RuntimeSession {
    run: Box<dyn crate::TtsModelRun>,
    state: RuntimeSessionState,
    chunk_steps: usize,
    stream: bool,
    generated_steps: usize,
    emitted_audio_steps: usize,
    emitted_samples: usize,
    events: Vec<SynthesisEvent>,
}

impl RuntimeSession {
    fn new(run: Box<dyn crate::TtsModelRun>, options: &SynthesisOptions) -> Self {
        Self {
            run,
            state: RuntimeSessionState::Running,
            chunk_steps: options.chunk_steps,
            stream: options.stream,
            generated_steps: 0,
            emitted_audio_steps: 0,
            emitted_samples: 0,
            events: Vec::new(),
        }
    }

    fn advance_until_finished(mut self) -> Result<SynthesisResult, TtsCoreError> {
        while self.state != RuntimeSessionState::Finished {
            let step = self.run.advance()?;
            self.on_model_step(step)?;
        }
        self.run.finish()
    }

    fn on_model_step(&mut self, step: ModelStep) -> Result<(), TtsCoreError> {
        if step.generated_steps > 0 {
            self.generated_steps += step.generated_steps;
            self.events.push(SynthesisEvent::CodecChunk {
                steps: step.generated_steps,
            });
            self.maybe_emit_audio(step.finished)?;
        }
        if step.finished {
            self.state = RuntimeSessionState::Finished;
            self.events.push(SynthesisEvent::Finished);
        }
        Ok(())
    }

    fn maybe_emit_audio(&mut self, finished: bool) -> Result<(), TtsCoreError> {
        if !self.stream {
            return Ok(());
        }
        let pending_steps = self
            .generated_steps
            .saturating_sub(self.emitted_audio_steps);
        if !should_emit_audio_chunk(pending_steps, self.chunk_steps, finished) {
            return Ok(());
        }
        let result = self.run.decode_audio()?;
        let delta = result.waveform_pcm[self.emitted_samples..].to_vec();
        self.events
            .push(SynthesisEvent::AudioChunk(crate::AudioChunk {
                pcm: delta,
                sample_rate: result.sample_rate,
                is_final: finished,
            }));
        self.emitted_audio_steps = self.generated_steps;
        self.emitted_samples = result.waveform_pcm.len();
        Ok(())
    }
}

pub struct TtsService {
    registry: Arc<ModelRegistry>,
}

impl TtsService {
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }

    pub fn synthesize(
        &self,
        model_id: &str,
        request: &SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult, TtsCoreError> {
        if request.text.trim().is_empty() {
            return Err(TtsCoreError::InvalidRequest {
                message: "text cannot be empty".to_string(),
            });
        }

        let executor = self
            .registry
            .get(model_id)
            .ok_or_else(|| TtsCoreError::UnknownModel {
                model_id: model_id.to_string(),
            })?;
        RuntimeSession::new(executor.start_run(request, options)?, options).advance_until_finished()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ModelCapabilities, TtsModelExecutor, TtsModelRun};

    struct MockExecutor;
    struct MockRun {
        steps_left: usize,
        sample: i16,
    }

    impl TtsModelRun for MockRun {
        fn advance(&mut self) -> Result<ModelStep, TtsCoreError> {
            if self.steps_left == 0 {
                return Ok(ModelStep {
                    generated_steps: 0,
                    finished: true,
                });
            }
            self.steps_left -= 1;
            Ok(ModelStep {
                generated_steps: 1,
                finished: self.steps_left == 0,
            })
        }

        fn decode_audio(&self) -> Result<SynthesisResult, TtsCoreError> {
            Ok(SynthesisResult {
                waveform_pcm: vec![self.sample],
                sample_rate: 24000,
            })
        }

        fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError> {
            Ok(SynthesisResult {
                waveform_pcm: vec![self.sample],
                sample_rate: 24000,
            })
        }
    }

    impl TtsModelExecutor for MockExecutor {
        fn family(&self) -> &'static str {
            "mock"
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }

        fn start_run(
            &self,
            request: &SynthesisRequest,
            _options: &SynthesisOptions,
        ) -> Result<Box<dyn TtsModelRun>, TtsCoreError> {
            Ok(Box::new(MockRun {
                steps_left: 2,
                sample: request.text.len() as i16,
            }))
        }
    }

    #[test]
    fn synthesize_routes_to_registered_adapter() {
        let mut registry = ModelRegistry::new();
        registry.register("mock", Arc::new(MockExecutor));
        let service = TtsService::new(registry);

        let request = SynthesisRequest {
            text: "hello".to_string(),
            language: None,
            speaker: None,
        };
        let result = service
            .synthesize("mock", &request, &SynthesisOptions::default())
            .expect("mock adapter should run");

        assert_eq!(result.waveform_pcm, vec![5]);
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn synthesize_fails_for_unknown_model() {
        let service = TtsService::new(ModelRegistry::new());
        let request = SynthesisRequest {
            text: "hello".to_string(),
            language: None,
            speaker: None,
        };

        let error = service
            .synthesize("missing", &request, &SynthesisOptions::default())
            .expect_err("missing model should fail");

        assert!(matches!(error, TtsCoreError::UnknownModel { .. }));
    }
}
