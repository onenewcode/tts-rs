use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use clap::Parser;
use tracing::info;
use tts_qwen::{BackendKind, CustomVoiceRequest, default_engine_config, default_session_config};

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
    #[arg(long)]
    pub backend: Option<String>,
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

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
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

pub fn run_from_args() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    init_logging(args.log_level);
    let total_started = Instant::now();
    std::fs::create_dir_all(&args.output_dir)?;

    let backend = parse_backend(args.backend.as_deref())?;
    info!(
        model_dir = %args.model_dir.display(),
        output_dir = %args.output_dir.display(),
        backend = %backend,
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
    let finished = tts_qwen::run_with_backend(
        backend,
        &args.model_dir,
        request,
        default_engine_config(args.chunk_steps, args.profiling),
        default_session_config(args.max_new_tokens, args.stream),
    )
    .map_err(|e| format!("tts inference failed: {e}"))?;

    tts_qwen::save_pcm_wav(&finished.waveform_pcm, &wav_path, finished.sample_rate)
        .map_err(|e| format!("failed to save wav: {e}"))?;
    info!(
        wav_path = %wav_path.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
}

fn parse_backend(selected: Option<&str>) -> Result<BackendKind, Box<dyn std::error::Error>> {
    let selected = selected.map(BackendKind::from_str).transpose()?;
    Ok(tts_qwen::resolve_backend(selected)?)
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

    #[test]
    fn args_parse_backend_as_string() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--model-dir",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "--text",
            "hello",
            "--backend",
            "flex",
        ])
        .expect("backend arg should parse");

        assert_eq!(args.backend.as_deref(), Some("flex"));
    }
}
