use burn::tensor::backend::Backend;

use crate::runners::talker::TalkerGenerator;
use crate::runtime::sampling::SamplingConfig;
use crate::session::output::{PendingAudio, StreamEvent};
use crate::session::types::CompiledRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    Full,
    AudioChunks,
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub streaming: StreamingMode,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::greedy(),
            streaming: StreamingMode::Full,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionHandle(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsSessionState {
    Created,
    Prefilled,
    Generating,
    Draining,
    Finished,
    Failed,
}

#[derive(Debug)]
pub struct TtsSession<B: Backend> {
    pub id: usize,
    pub state: TtsSessionState,
    pub config: SessionConfig,
    pub compiled: CompiledRequest<B>,
    pub talker: Option<TalkerGenerator<B>>,
    pub pending_audio: PendingAudio,
    pub queued_events: Vec<StreamEvent>,
}
