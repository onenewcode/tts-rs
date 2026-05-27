use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::Backend;
use clap::{Parser, ValueEnum};
use tracing::info;
use tts_qwen::{CustomVoiceRequest, LocalInferenceCore, LocalInferenceOptions, QwenTtsAdapter};

#[derive(Debug, Parser)]
#[command(name = "tts_cli")]
pub struct Args {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub text: String,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long)]
    pub speaker: Option<String>,
    #[arg(long, value_enum, default_value_t = ModelKind::QwenTts)]
    pub model_kind: ModelKind,
    #[arg(long, value_enum)]
    pub backend: Option<BackendKind>,
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_tracing_level(self) -> tracing::Level {
        match self {
            Self::Error => tracing::Level::ERROR,
            Self::Warn => tracing::Level::WARN,
            Self::Info => tracing::Level::INFO,
            Self::Debug => tracing::Level::DEBUG,
            Self::Trace => tracing::Level::TRACE,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum BackendKind {
    #[cfg(feature = "backend-flex")]
    Flex,
    #[cfg(feature = "backend-ndarray")]
    Ndarray,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ModelKind {
    QwenTts,
}

pub fn run_from_args() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    init_logging(args.log_level);
    let total_started = Instant::now();
    std::fs::create_dir_all(&args.output_dir)?;
    let backend = resolve_backend(args.backend)?;
    info!(
        model_dir = %args.model_dir.display(),
        output_dir = %args.output_dir.display(),
        model_kind = %args.model_kind.label(),
        backend = %backend.label(),
        max_new_tokens = args.max_new_tokens,
        language = args.language.as_deref().unwrap_or("Auto"),
        speaker = args.speaker.as_deref().unwrap_or(""),
        "starting tts generation"
    );

    let request = CustomVoiceRequest {
        text: args.text.clone(),
        language: args.language.clone(),
        speaker: args.speaker.clone(),
    };
    let wav_path = args.output_dir.join("0000.wav");
    let output = match backend {
        #[cfg(feature = "backend-flex")]
        BackendKind::Flex => run_with_backend::<burn::backend::Flex>(&args, &request, &wav_path)?,
        #[cfg(feature = "backend-ndarray")]
        BackendKind::Ndarray => {
            run_with_backend::<burn::backend::NdArray>(&args, &request, &wav_path)?
        }
    };
    info!(
        wav_path = %wav_path.display(),
        sample_rate = output.sample_rate,
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
}

fn run_with_backend<B>(
    args: &Args,
    request: &CustomVoiceRequest,
    wav_path: &std::path::Path,
) -> Result<tts_qwen::Qwen3TtsInferOutput<B>, Box<dyn std::error::Error>>
where
    B: Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    let core = match args.model_kind {
        ModelKind::QwenTts => {
            LocalInferenceCore::<B, QwenTtsAdapter<B>>::load(&args.model_dir, &device)
                .map_err(|e| format!("failed to load inference core: {e}"))?
        }
    };
    core.infer_to_file(
        request,
        &LocalInferenceOptions {
            max_new_tokens: args.max_new_tokens,
            ..LocalInferenceOptions::default()
        },
        wav_path,
    )
    .map(|run| run.output)
    .map_err(|e| format!("tts inference failed: {e}").into())
}

fn resolve_backend(
    selected: Option<BackendKind>,
) -> Result<BackendKind, Box<dyn std::error::Error>> {
    let available = available_backends();
    match (selected, available.as_slice()) {
        (Some(backend), _) => Ok(backend),
        (None, [backend]) => Ok(*backend),
        (None, _) => Err(format!(
            "multiple backends are compiled in; pass --backend one of: {}",
            available
                .iter()
                .map(|backend| backend.label())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into()),
    }
}

fn available_backends() -> Vec<BackendKind> {
    let mut backends = Vec::new();
    #[cfg(feature = "backend-flex")]
    backends.push(BackendKind::Flex);
    #[cfg(feature = "backend-ndarray")]
    backends.push(BackendKind::Ndarray);
    backends
}

impl BackendKind {
    fn label(&self) -> &'static str {
        match self {
            #[cfg(feature = "backend-flex")]
            Self::Flex => "flex",
            #[cfg(feature = "backend-ndarray")]
            Self::Ndarray => "ndarray",
        }
    }
}

impl ModelKind {
    fn label(&self) -> &'static str {
        match self {
            Self::QwenTts => "qwen-tts",
        }
    }
}

fn init_logging(level: LogLevel) {
    let _ = tracing_subscriber::fmt()
        .with_max_level(level.as_tracing_level())
        .with_target(false)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parse_required_fields_with_defaults() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--model-dir",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "--text",
            "hello",
        ])
        .expect("minimal args should parse");

        assert_eq!(
            args.model_dir,
            PathBuf::from("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        );
        assert_eq!(args.text, "hello");
        assert_eq!(args.language, None);
        assert_eq!(args.speaker, None);
        assert_eq!(args.model_kind, ModelKind::QwenTts);
        assert_eq!(args.backend, None);
        assert_eq!(args.output_dir, PathBuf::from("output"));
        assert_eq!(args.max_new_tokens, 256);
        assert_eq!(args.log_level, LogLevel::Info);
    }

    #[test]
    fn args_parse_optional_generation_fields() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--model-dir",
            "model",
            "--text",
            "你好",
            "--language",
            "Chinese",
            "--speaker",
            "Vivian",
            "--model-kind",
            "qwen-tts",
            "--backend",
            available_backends()[0].label(),
            "--output-dir",
            ".",
            "--max-new-tokens",
            "64",
            "--log-level",
            "debug",
        ])
        .expect("full args should parse");

        assert_eq!(args.language.as_deref(), Some("Chinese"));
        assert_eq!(args.speaker.as_deref(), Some("Vivian"));
        assert_eq!(args.model_kind, ModelKind::QwenTts);
        assert_eq!(args.output_dir, PathBuf::from("."));
        assert_eq!(args.max_new_tokens, 64);
        assert_eq!(args.log_level, LogLevel::Debug);
    }

    #[test]
    fn resolve_backend_defaults_when_only_one_backend_is_compiled() {
        let available = available_backends();
        if available.len() == 1 {
            assert_eq!(resolve_backend(None).unwrap(), available[0]);
        }
    }

    #[cfg(all(feature = "backend-flex", feature = "backend-ndarray"))]
    #[test]
    fn resolve_backend_requires_explicit_choice_when_multiple_backends_are_compiled() {
        let err = resolve_backend(None).expect_err("backend choice should be required");
        assert!(
            err.to_string()
                .contains("multiple backends are compiled in"),
            "unexpected error: {err}"
        );
    }
}
