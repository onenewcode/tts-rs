use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ServiceError {
    #[error("session step called after terminal state")]
    StepAfterTerminal,
    #[error("session finish called before terminal state")]
    FinishBeforeTerminal,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum InferError<E> {
    #[error("model error: {0}")]
    Model(E),
    #[error(transparent)]
    Service(ServiceError),
}
