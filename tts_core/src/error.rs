use thiserror::Error;

#[derive(Debug, Error)]
pub enum TtsCoreError {
    #[error("unknown model id `{model_id}`")]
    UnknownModel { model_id: String },
    #[error("adapter `{model_type}` failed: {message}")]
    Adapter {
        model_type: &'static str,
        message: String,
    },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("configuration error: {message}")]
    Config { message: String },
}
