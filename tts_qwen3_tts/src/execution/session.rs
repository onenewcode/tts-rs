use super::error::{InferError, ServiceError};
use tts_infer::PcmAudio;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStep {
    Advanced,
    Finished,
}

pub trait ModelSession {
    type Error;

    fn step(&mut self) -> Result<SessionStep, Self::Error>;
    fn finish(self) -> Result<PcmAudio, Self::Error>;
}

#[derive(Debug)]
pub struct EngineSession<S> {
    inner: S,
    state: EngineSessionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineSessionState {
    Running,
    TerminalReached,
}

impl<S> EngineSession<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            state: EngineSessionState::Running,
        }
    }
}

impl<S> EngineSession<S>
where
    S: ModelSession,
{
    pub fn step(&mut self) -> Result<SessionStep, InferError<S::Error>> {
        if self.state == EngineSessionState::TerminalReached {
            return Err(InferError::Service(ServiceError::StepAfterTerminal));
        }

        let step = self.inner.step().map_err(InferError::Model)?;
        if step == SessionStep::Finished {
            self.state = EngineSessionState::TerminalReached;
        }

        Ok(step)
    }

    pub fn finish(self) -> Result<PcmAudio, InferError<S::Error>> {
        if self.state != EngineSessionState::TerminalReached {
            return Err(InferError::Service(ServiceError::FinishBeforeTerminal));
        }

        self.inner.finish().map_err(InferError::Model)
    }
}
