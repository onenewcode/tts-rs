use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::Backend;
use clap::{Parser, ValueEnum};
use tracing::info;
use tts_qwen::{
    CustomVoiceRequest, EngineConfig, ProfilingConfig, QwenTtsEngine, SamplingConfig,
    SessionConfig, StreamingMode,
};

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
    #[arg(long, value_enum)]
    pub backend: Option<BackendKind>,
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    #[arg(long, default_value_t = 8)]
    pub chunk_steps: usize,
    #[arg(long)]
    pub stream: bool,
    #[arg(long)]
    pub profiling: bool,
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
        backend = %backend.label(),
        max_new_tokens = args.max_new_tokens,
        chunk_steps = args.chunk_steps,
        stream = args.stream,
        profiling = args.profiling,
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
    match backend {
        #[cfg(feature = "backend-flex")]
        BackendKind::Flex => run_with_backend::<burn::backend::Flex>(&args, request, &wav_path)?,
        #[cfg(feature = "backend-ndarray")]
        BackendKind::Ndarray => {
            run_with_backend::<burn::backend::NdArray>(&args, request, &wav_path)?
        }
    }
    info!(
        wav_path = %wav_path.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
}

fn run_with_backend<B>(
    args: &Args,
    request: CustomVoiceRequest,
    wav_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    let mut engine = QwenTtsEngine::<B>::load(
        &args.model_dir,
        &device,
        EngineConfig {
            codec_chunk_steps: args.chunk_steps,
            profiling: ProfilingConfig {
                enabled: args.profiling,
                per_step: args.profiling,
                stage_summary: true,
                log_topk: 8,
            },
            ..EngineConfig::default()
        },
    )
    .map_err(|e| format!("failed to load engine: {e}"))?;
    let handle = engine
        .start_session(
            request,
            SessionConfig {
                max_new_tokens: args.max_new_tokens,
                sampling: SamplingConfig::greedy(),
                streaming: if args.stream {
                    StreamingMode::AudioChunks
                } else {
                    StreamingMode::Full
                },
            },
        )
        .map_err(|e| format!("failed to start session: {e}"))?;

    let finished = if args.stream {
        loop {
            let outcome = engine.step(handle).map_err(|e| format!("step failed: {e}"))?;
            let events = engine
                .drain_events(handle)
                .map_err(|e| format!("drain failed: {e}"))?;
            for event in events {
                info!(event = ?event, "stream event");
            }
            if matches!(outcome, tts_qwen::StepOutcome::Finished) {
                break engine
                    .finish_session(handle)
                    .map_err(|e| format!("tts inference failed: {e}"))?;
            }
        }
    } else {
        engine
            .run_to_end(handle)
            .map_err(|e| format!("tts inference failed: {e}"))?
    };

    tts_qwen::save_pcm_wav(&finished.waveform_pcm, wav_path, finished.sample_rate)
        .map_err(|e| format!("failed to save wav: {e}"))?;
    Ok(())
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
        assert_eq!(args.output_dir, PathBuf::from("output"));
        assert_eq!(args.max_new_tokens, 256);
        assert_eq!(args.chunk_steps, 8);
        assert!(!args.stream);
        assert!(!args.profiling);
        assert_eq!(args.log_level, LogLevel::Info);
    }
}
