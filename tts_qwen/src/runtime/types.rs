use crate::profiling::ProfilingConfig;
use tts_core::runtime::sampling::SamplingConfig;

#[derive(Debug, Clone)]
pub(crate) struct EngineConfig {
    pub profiling: ProfilingConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct RunConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunStep {
    pub generated_steps: usize,
    pub finished: bool,
}
