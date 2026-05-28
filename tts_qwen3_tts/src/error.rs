use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen3TtsLoadError {
    #[error("failed to load qwen3 tts config from {path}: {source}")]
    Config {
        path: PathBuf,
        #[source]
        source: burn::config::ConfigError,
    },
    #[error("failed to read checkpoint from {path}: {source}")]
    Store {
        path: PathBuf,
        #[source]
        source: burn_store::SafetensorsStoreError,
    },
    #[error("checkpoint loaded but {unused} tensors were left unused")]
    UnusedTensors { unused: usize },
    #[error("failed to read package manifest {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse package manifest {path}: {source}")]
    ManifestParse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("failed to read compiler config {path}: {source}")]
    CompilerConfigIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse compiler config {path}: {source}")]
    CompilerConfigParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to load tokenizer from {path}: {source}")]
    Tokenizer {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },
    #[error("backend `{backend}` is not compiled in")]
    UnavailableBackend { backend: String },
    #[error("invalid package manifest: {message}")]
    InvalidManifest { message: String },
}

#[derive(Debug, Error)]
pub enum Qwen3TtsInferenceError {
    #[error("invalid inference input: {message}")]
    InvalidInput { message: String },
    #[error("model runtime load failed: {message}")]
    RuntimeLoad { message: String },
    #[error("unsupported activation function: {name}")]
    UnsupportedActivation { name: String },
    #[error("unsupported rope configuration: {message}")]
    UnsupportedRope { message: String },
    #[error("tokenizer error: {source}")]
    Tokenizer {
        #[from]
        source: tokenizers::Error,
    },
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read tensor data: {message}")]
    TensorRead { message: String },
}

#[derive(Debug, Error)]
pub enum Qwen3TtsError {
    #[error(transparent)]
    Load(#[from] Qwen3TtsLoadError),
    #[error(transparent)]
    Infer(#[from] tts_infer::InferError<Qwen3TtsInferenceError>),
    #[error(transparent)]
    Inference(#[from] Qwen3TtsInferenceError),
}

pub type QwenTtsInferenceError = Qwen3TtsInferenceError;
