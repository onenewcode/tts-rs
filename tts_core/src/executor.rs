use crate::{SynthesisOptions, SynthesisRequest, SynthesisResult, TtsCoreError};

#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    pub supports_streaming: bool,
    pub supports_custom_voice: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelStep {
    pub generated_steps: usize,
    pub finished: bool,
}

pub trait TtsModelRun: Send {
    fn advance(&mut self) -> Result<ModelStep, TtsCoreError>;

    fn decode_audio(&self) -> Result<SynthesisResult, TtsCoreError>;

    fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError>;
}

pub trait TtsModelExecutor: Send + Sync {
    fn family(&self) -> &'static str;

    fn capabilities(&self) -> ModelCapabilities;

    fn start_run(
        &self,
        request: &SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsModelRun>, TtsCoreError>;
}
