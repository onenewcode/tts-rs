use std::path::PathBuf;

use thiserror::Error;
use tts_error::DiagnosticError;

#[derive(Debug, Error)]
pub enum Qwen3TtsLoadError {
    #[error("failed to load qwen3 tts config from {path}: {source}")]
    Config {
        path: PathBuf,
        #[source]
        source: burn::config::ConfigError,
    },
    #[error("failed to read model weights from {path}: {source}")]
    Store {
        path: PathBuf,
        #[source]
        source: burn_store::SafetensorsStoreError,
    },
    #[error("model weights loaded but {unused} tensors were left unused")]
    UnusedTensors { unused: usize },
    #[error("unsupported runtime dtype {requested}; use f16, f32, or bf16")]
    UnsupportedDType { requested: String },
    #[error("failed to initialize runtime dtype {requested}: {message}")]
    RuntimeDType { requested: String, message: String },
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
    #[error("invalid compiler config: {message}")]
    InvalidCompilerConfig { message: String },
    #[error("invalid package manifest: {message}")]
    InvalidManifest { message: String },
    #[error("invalid model directory: {message}")]
    InvalidModelDir { message: String },
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
    #[error("failed to decode audio input: {message}")]
    AudioDecode { message: String },
}

#[derive(Debug, Error)]
pub enum Qwen3TtsError {
    #[error(transparent)]
    Framework(#[from] DiagnosticError),
    #[error(transparent)]
    Load(#[from] Qwen3TtsLoadError),
    #[error(transparent)]
    Infer(#[from] crate::execution::error::InferError<Qwen3TtsInferenceError>),
    #[error(transparent)]
    Inference(#[from] Qwen3TtsInferenceError),
}

pub type QwenTtsInferenceError = Qwen3TtsInferenceError;

impl From<Qwen3TtsLoadError> for DiagnosticError {
    fn from(value: Qwen3TtsLoadError) -> Self {
        match value {
            Qwen3TtsLoadError::Config { path, source } => invalid_artifact(
                "qwen3.load.config",
                "failed to load runtime config",
                path,
                source.to_string(),
            ),
            Qwen3TtsLoadError::Store { path, source } => invalid_artifact(
                "qwen3.load.store",
                "failed to read safetensors weights",
                path,
                source.to_string(),
            ),
            Qwen3TtsLoadError::UnusedTensors { unused } => DiagnosticError::invalid_argument(
                "qwen3.load.unused_tensors",
                format!("model weights loaded but {unused} tensors were left unused"),
            )
            .with_context("unused", unused.to_string()),
            Qwen3TtsLoadError::UnsupportedDType { requested } => DiagnosticError::invalid_argument(
                "qwen3.load.unsupported_dtype",
                format!("unsupported runtime dtype {requested}; use f16, f32, or bf16"),
            )
            .with_context("requested", requested),
            Qwen3TtsLoadError::RuntimeDType { requested, message } => {
                DiagnosticError::invalid_argument(
                    "qwen3.load.runtime_dtype",
                    format!("failed to initialize runtime dtype {requested}: {message}"),
                )
                .with_context("requested", requested)
                .with_context("message", message)
            }
            Qwen3TtsLoadError::Io { path, source } => io_artifact(
                "qwen3.load.manifest_io",
                "failed to read package manifest",
                path,
                source,
            ),
            Qwen3TtsLoadError::ManifestParse { path, source } => invalid_artifact(
                "qwen3.load.manifest_parse",
                "failed to parse package manifest",
                path,
                source.to_string(),
            ),
            Qwen3TtsLoadError::CompilerConfigIo { path, source } => io_artifact(
                "qwen3.load.compiler_config_io",
                "failed to read compiler config",
                path,
                source,
            ),
            Qwen3TtsLoadError::CompilerConfigParse { path, source } => invalid_artifact(
                "qwen3.load.compiler_config_parse",
                "failed to parse compiler config",
                path,
                source.to_string(),
            ),
            Qwen3TtsLoadError::Tokenizer { path, source } => invalid_artifact(
                "qwen3.load.tokenizer",
                "failed to load tokenizer",
                path,
                source.to_string(),
            ),
            Qwen3TtsLoadError::InvalidCompilerConfig { message } => {
                DiagnosticError::invalid_argument(
                    "qwen3.load.invalid_compiler_config",
                    format!("invalid compiler config: {message}"),
                )
                .with_context("message", message)
            }
            Qwen3TtsLoadError::InvalidManifest { message } => DiagnosticError::invalid_argument(
                "qwen3.load.invalid_manifest",
                format!("invalid package manifest: {message}"),
            )
            .with_context("message", message),
            Qwen3TtsLoadError::InvalidModelDir { message } => DiagnosticError::invalid_argument(
                "qwen3.load.invalid_model_dir",
                format!("invalid model directory: {message}"),
            )
            .with_context("message", message),
        }
    }
}

fn invalid_artifact(
    code: &'static str,
    message: &'static str,
    path: PathBuf,
    source: String,
) -> DiagnosticError {
    DiagnosticError::invalid_argument(code, format!("{message}: {}", path.display()))
        .with_context("path", path.display().to_string())
        .with_context("source", source)
}

fn io_artifact(
    code: &'static str,
    message: &'static str,
    path: PathBuf,
    source: std::io::Error,
) -> DiagnosticError {
    let diagnostic = if source.kind() == std::io::ErrorKind::NotFound {
        DiagnosticError::not_found(code, format!("{message}: {}", path.display()))
    } else {
        DiagnosticError::invalid_argument(code, format!("{message}: {}", path.display()))
    };

    diagnostic
        .with_context("path", path.display().to_string())
        .with_context("source", source.to_string())
}
