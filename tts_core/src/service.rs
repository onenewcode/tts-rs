use std::sync::Arc;

use crate::{
    ModelRegistry, SessionStep, SynthesisOptions, SynthesisRequest, SynthesisResult, TtsCoreError,
};

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

        let adapter = self
            .registry
            .get(model_id)
            .ok_or_else(|| TtsCoreError::UnknownModel {
                model_id: model_id.to_string(),
            })?;

        let mut session = adapter.start_session(request, options)?;
        loop {
            if matches!(session.step()?, SessionStep::Finished) {
                break;
            }
            let _ = session.drain_events()?;
        }
        session.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ModelCapabilities, SessionStep, SynthesisEvent, TtsModelAdapter, TtsModelSession};

    struct MockAdapter;
    struct MockSession {
        done: bool,
        sample: i16,
    }

    impl TtsModelSession for MockSession {
        fn step(&mut self) -> Result<SessionStep, TtsCoreError> {
            if self.done {
                Ok(SessionStep::Finished)
            } else {
                self.done = true;
                Ok(SessionStep::Running)
            }
        }

        fn drain_events(&mut self) -> Result<Vec<SynthesisEvent>, TtsCoreError> {
            Ok(vec![SynthesisEvent::CodecChunk { steps: 1 }])
        }

        fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError> {
            Ok(SynthesisResult {
                waveform_pcm: vec![self.sample],
                sample_rate: 24000,
            })
        }
    }

    impl TtsModelAdapter for MockAdapter {
        fn family(&self) -> &'static str {
            "mock"
        }

        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities::default()
        }

        fn start_session(
            &self,
            request: &SynthesisRequest,
            _options: &SynthesisOptions,
        ) -> Result<Box<dyn TtsModelSession>, TtsCoreError> {
            Ok(Box::new(MockSession {
                done: false,
                sample: request.text.len() as i16,
            }))
        }
    }

    #[test]
    fn synthesize_routes_to_registered_adapter() {
        let mut registry = ModelRegistry::new();
        registry.register("mock", Arc::new(MockAdapter));
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
