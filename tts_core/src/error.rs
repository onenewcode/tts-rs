use thiserror::Error;

#[derive(Debug, Error)]
pub enum TtsCoreError {
    #[error("unknown model id `{model_id}`")]
    UnknownModel { model_id: String },
    #[error("executor `{family}` failed: {message}")]
    Executor {
        family: &'static str,
        message: String,
    },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("configuration error: {message}")]
    Config { message: String },
}
