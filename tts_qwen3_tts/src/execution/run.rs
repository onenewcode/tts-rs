use super::error::InferError;
use super::session::{EngineSession, ModelSession, SessionStep};
use tts_core::PcmAudio;

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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{Engine, LoadedModel};
    use crate::execution::error::{InferError, ServiceError};
    use crate::execution::session::{ModelSession, SessionStep};
    use tts_core::PcmAudio;

    #[derive(Clone)]
    struct MockLoadedModel {
        finish_after: usize,
        starts: Rc<Cell<usize>>,
    }

    impl MockLoadedModel {
        fn new(finish_after: usize) -> Self {
            Self {
                finish_after,
                starts: Rc::new(Cell::new(0)),
            }
        }

        fn starts(&self) -> usize {
            self.starts.get()
        }
    }

    impl LoadedModel for MockLoadedModel {
        type Request = &'static str;
        type RunOptions = usize;
        type Session = MockSession;
        type Error = &'static str;

        fn start_session(
            &self,
            request: Self::Request,
            options: Self::RunOptions,
        ) -> Result<Self::Session, Self::Error> {
            self.starts.set(self.starts.get() + 1);
            Ok(MockSession::new(
                request.len() as i16,
                self.finish_after.max(options),
            ))
        }
    }

    struct MockSession {
        sample: i16,
        finish_after: usize,
        steps: usize,
    }

    impl MockSession {
        fn new(sample: i16, finish_after: usize) -> Self {
            Self {
                sample,
                finish_after,
                steps: 0,
            }
        }
    }

    impl ModelSession for MockSession {
        type Error = &'static str;

        fn step(&mut self) -> Result<SessionStep, Self::Error> {
            self.steps += 1;
            if self.steps >= self.finish_after {
                Ok(SessionStep::Finished)
            } else {
                Ok(SessionStep::Advanced)
            }
        }

        fn finish(self) -> Result<PcmAudio, Self::Error> {
            Ok(PcmAudio {
                pcm_i16: vec![self.sample],
                sample_rate: 24_000,
                channels: 1,
            })
        }
    }

    struct FailingLoadedModel;

    impl LoadedModel for FailingLoadedModel {
        type Request = ();
        type RunOptions = ();
        type Session = FailingSession;
        type Error = &'static str;

        fn start_session(
            &self,
            _request: Self::Request,
            _options: Self::RunOptions,
        ) -> Result<Self::Session, Self::Error> {
            Ok(FailingSession)
        }
    }

    struct FailingSession;

    impl ModelSession for FailingSession {
        type Error = &'static str;

        fn step(&mut self) -> Result<SessionStep, Self::Error> {
            Err("step failed")
        }

        fn finish(self) -> Result<PcmAudio, Self::Error> {
            Err("finish failed")
        }
    }

    #[test]
    fn synthesize_runs_session_until_finish_and_returns_pcm_audio() {
        let model = MockLoadedModel::new(2);
        let engine = Engine::new(model.clone());

        let audio = engine
            .synthesize("hello", 1)
            .expect("session should finish");

        assert_eq!(model.starts(), 1);
        assert_eq!(audio.pcm_i16, vec![5]);
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.channels, 1);
    }

    #[test]
    fn session_step_rejects_calls_after_terminal_state() {
        let model = MockLoadedModel::new(1);
        let engine = Engine::new(model);
        let mut session = engine.start_session("hi", 1).expect("session should start");

        assert_eq!(session.step().unwrap(), SessionStep::Finished);
        let error = session
            .step()
            .expect_err("terminal session should reject more steps");

        assert_eq!(error, InferError::Service(ServiceError::StepAfterTerminal));
    }

    #[test]
    fn session_finish_rejects_calls_before_terminal_state() {
        let model = MockLoadedModel::new(2);
        let engine = Engine::new(model);
        let session = engine.start_session("hi", 1).expect("session should start");

        let error = session
            .finish()
            .expect_err("non-terminal session should reject finish");

        assert_eq!(
            error,
            InferError::Service(ServiceError::FinishBeforeTerminal)
        );
    }

    #[test]
    fn session_finish_returns_audio_after_terminal_state() {
        let model = MockLoadedModel::new(1);
        let engine = Engine::new(model);
        let mut session = engine.start_session("hi", 1).expect("session should start");

        assert_eq!(session.step().unwrap(), SessionStep::Finished);
        let audio = session.finish().expect("terminal session should finish");

        assert_eq!(audio.pcm_i16, vec![2]);
    }

    #[test]
    fn model_step_errors_are_wrapped() {
        let engine = Engine::new(FailingLoadedModel);

        let error = engine
            .synthesize((), ())
            .expect_err("step failure should bubble up");

        assert_eq!(error, InferError::Model("step failed"));
    }
}
