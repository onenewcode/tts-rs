use crate::{
    SessionStep, SynthesisEvent, SynthesisOptions, SynthesisRequest, SynthesisResult, TtsCoreError,
};

#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    pub supports_streaming: bool,
    pub supports_custom_voice: bool,
}

pub trait TtsModelSession: Send {
    fn step(&mut self) -> Result<SessionStep, TtsCoreError>;

    fn drain_events(&mut self) -> Result<Vec<SynthesisEvent>, TtsCoreError>;

    fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError>;
}

pub trait TtsModelAdapter: Send + Sync {
    fn family(&self) -> &'static str;

    fn capabilities(&self) -> ModelCapabilities;

    fn start_session(
        &self,
        request: &SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsModelSession>, TtsCoreError>;
}
