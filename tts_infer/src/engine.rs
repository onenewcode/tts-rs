use crate::{EngineSession, InferError, ModelSession, PcmAudio, SessionStep};

pub trait LoadedModel {
    type Request;
    type RunOptions;
    type Session: ModelSession<Error = Self::Error>;
    type Error;

    fn start_session(
        &self,
        request: Self::Request,
        options: Self::RunOptions,
    ) -> Result<Self::Session, Self::Error>;
}

#[derive(Debug, Clone)]
pub struct Engine<M> {
    model: M,
}

impl<M> Engine<M>
where
    M: LoadedModel,
{
    pub fn new(model: M) -> Self {
        Self { model }
    }
    /// TODO 设置合理吗，会一直空转吗
    pub fn synthesize(
        &self,
        request: M::Request,
        options: M::RunOptions,
    ) -> Result<PcmAudio, InferError<M::Error>> {
        let mut session = self.start_session(request, options)?;
        loop {
            if session.step()? == SessionStep::Finished {
                return session.finish();
            }
        }
    }

    pub fn start_session(
        &self,
        request: M::Request,
        options: M::RunOptions,
    ) -> Result<EngineSession<M::Session>, InferError<M::Error>> {
        let session = self
            .model
            .start_session(request, options)
            .map_err(InferError::Model)?;
        Ok(EngineSession::new(session))
    }
}
