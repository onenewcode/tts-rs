use std::path::PathBuf;
use std::time::Instant;

use burn::backend::Flex;
use clap::{Parser, ValueEnum};
use tracing::info;
use tts_qwen::{CustomVoiceRequest, Qwen3TtsPipeline, Qwen3TtsSynthesisOptions};

type Backend = Flex;

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

pub fn run_from_args() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    init_logging(args.log_level);
    let total_started = Instant::now();
    std::fs::create_dir_all(&args.output_dir)?;
    info!(
        model_dir = %args.model_dir.display(),
        output_dir = %args.output_dir.display(),
        max_new_tokens = args.max_new_tokens,
        language = args.language.as_deref().unwrap_or("Auto"),
        speaker = args.speaker.as_deref().unwrap_or(""),
        "starting tts generation"
    );

    let device = Default::default();
    let pipeline = Qwen3TtsPipeline::<Backend>::load(&args.model_dir, &device)
        .map_err(|e| format!("failed to load pipeline: {e}"))?;

    let request = CustomVoiceRequest {
        text: args.text.clone(),
        language: args.language.clone(),
        speaker: args.speaker.clone(),
    };
    let wav_path = args.output_dir.join("0000.wav");
    let output = pipeline
        .synthesize_to_wav(
            &request,
            &Qwen3TtsSynthesisOptions {
                max_new_tokens: args.max_new_tokens,
                ..Qwen3TtsSynthesisOptions::default()
            },
            &wav_path,
        )
        .map_err(|e| format!("tts synthesis failed: {e}"))?;
    info!(
        wav_path = %wav_path.display(),
        sample_rate = output.sample_rate,
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
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
        assert_eq!(args.output_dir, PathBuf::from("."));
        assert_eq!(args.max_new_tokens, 64);
        assert_eq!(args.log_level, LogLevel::Debug);
    }
}
