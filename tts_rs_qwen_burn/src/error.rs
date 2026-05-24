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
    #[error("unable to find a Qwen model directory under {root}")]
    ModelDirNotFound { root: PathBuf },
    #[error("filesystem error while scanning {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Error)]
pub enum Qwen3TtsVerifyError {
    #[error("failed to read source checkpoint from {path}: {source}")]
    Store {
        path: PathBuf,
        #[source]
        source: burn_store::SafetensorsStoreError,
    },
    #[error("exported model and source checkpoint do not expose the same tensor keys")]
    KeySetMismatch {
        missing_in_export: Vec<String>,
        missing_in_source: Vec<String>,
    },
    #[error("tensor mismatch for {path}: {reason}")]
    TensorMismatch { path: String, reason: String },
    #[error("filesystem error while writing {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json serialization error for {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}
