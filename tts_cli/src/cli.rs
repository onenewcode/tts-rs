use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{ArgAction, Args as ClapArgs, Parser, Subcommand, ValueEnum};
use tracing::info;
use tts_qwen3_tts::{
    BaseRequest, CustomVoiceRequest, LanguageSelection, Qwen3TtsBackend, Qwen3TtsEngine,
    Qwen3TtsEngineConfig, Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions,
    QwenRequest, SamplingConfig,
};

#[derive(Debug, Parser)]
#[command(name = "tts_cli")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
    #[arg(long, value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,
}

pub type Args = Cli;

#[derive(Debug, Subcommand)]
pub enum Command {
    Synthesize(SynthesizeArgs),
}

#[derive(Debug, ClapArgs)]
pub struct SynthesizeArgs {
    #[command(subcommand)]
    pub profile: ProfileCommand,
}

#[derive(Debug, Subcommand)]
pub enum ProfileCommand {
    Base(BaseSynthesizeArgs),
    CustomVoice(CustomVoiceSynthesizeArgs),
}

#[derive(Debug, Clone, ClapArgs)]
pub struct SharedSynthesizeArgs {
    #[arg(long, required_unless_present = "manifest")]
    pub model_dir: Option<PathBuf>,
    #[arg(
        long,
        conflicts_with = "model_dir",
        required_unless_present = "model_dir"
    )]
    pub manifest: Option<PathBuf>,
    #[arg(long)]
    pub text: String,
    #[arg(long, default_value = "auto")]
    pub language: String,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, value_enum)]
    pub backend: Option<CliBackend>,
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    #[arg(long, value_enum, default_value_t = CliSampling::Greedy)]
    pub sampling: CliSampling,
    #[arg(long)]
    pub profiling: bool,
    #[arg(long)]
    pub profiling_per_step: bool,
    #[arg(long = "profiling-stage-summary", action = ArgAction::SetTrue, default_value_t = true)]
    pub profiling_stage_summary: bool,
    #[arg(long = "no-profiling-stage-summary", action = ArgAction::SetTrue)]
    pub no_profiling_stage_summary: bool,
    #[arg(long, default_value_t = 8)]
    pub profiling_log_topk: usize,
}

#[derive(Debug, Clone, ClapArgs)]
pub struct BaseSynthesizeArgs {
    #[command(flatten)]
    pub shared: SharedSynthesizeArgs,
}

#[derive(Debug, Clone, ClapArgs)]
pub struct CustomVoiceSynthesizeArgs {
    #[command(flatten)]
    pub shared: SharedSynthesizeArgs,
    #[arg(long)]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum CliBackend {
    Flex,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Webgpu,
}

impl CliBackend {
    fn to_backend(self) -> Qwen3TtsBackend {
        match self {
            Self::Flex => Qwen3TtsBackend::Flex,
            Self::Wgpu => Qwen3TtsBackend::Wgpu,
            Self::Cuda => Qwen3TtsBackend::Cuda,
            Self::Rocm => Qwen3TtsBackend::Rocm,
            Self::Metal => Qwen3TtsBackend::Metal,
            Self::Vulkan => Qwen3TtsBackend::Vulkan,
            Self::Webgpu => Qwen3TtsBackend::WebGpu,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum CliSampling {
    Greedy,
}

impl CliSampling {
    fn to_sampling(self) -> SamplingConfig {
        match self {
            Self::Greedy => SamplingConfig::greedy(),
        }
    }
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

    match args.command {
        Command::Synthesize(command) => match command.profile {
            ProfileCommand::Base(base) => run_synthesis(
                &base.shared,
                QwenRequest::Base(BaseRequest {
                    text: base.shared.text.clone(),
                    language: parse_language(&base.shared.language),
                }),
                total_started,
            ),
            ProfileCommand::CustomVoice(custom_voice) => run_synthesis(
                &custom_voice.shared,
                QwenRequest::CustomVoice(CustomVoiceRequest {
                    text: custom_voice.shared.text.clone(),
                    language: parse_language(&custom_voice.shared.language),
                    speaker: custom_voice.speaker.clone(),
                }),
                total_started,
            ),
        },
    }
}

fn run_synthesis(
    shared: &SharedSynthesizeArgs,
    request: QwenRequest,
    total_started: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = shared
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let package_source = package_source(shared)?;
    let engine = Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: package_source,
        backend: shared
            .backend
            .map(CliBackend::to_backend)
            .unwrap_or(Qwen3TtsBackend::Flex),
        profiling: Qwen3TtsProfilingConfig {
            enabled: shared.profiling,
            per_step: shared.profiling_per_step,
            stage_summary: resolve_stage_summary(shared),
            log_topk: shared.profiling_log_topk,
        },
    })?;
    let options = Qwen3TtsRunOptions {
        max_new_tokens: shared.max_new_tokens,
        sampling: shared.sampling.to_sampling(),
    };

    info!(
        source = %input_source_display(shared).display(),
        output = %shared.output.display(),
        backend = ?shared.backend,
        max_new_tokens = shared.max_new_tokens,
        profiling = shared.profiling,
        language = %shared.language,
        "starting tts generation"
    );

    let audio = engine.synthesize(request, options)?;
    audio.save_wav(&shared.output)?;

    info!(
        wav_path = %shared.output.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
}

fn package_source(shared: &SharedSynthesizeArgs) -> Result<Qwen3TtsPackageSource, std::io::Error> {
    match (&shared.model_dir, &shared.manifest) {
        (Some(path), None) => Ok(Qwen3TtsPackageSource::ModelDir(path.clone())),
        (None, Some(path)) => Ok(Qwen3TtsPackageSource::ManifestPath(path.clone())),
        (None, None) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pass either --model-dir or --manifest",
        )),
        (Some(_), Some(_)) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pass only one of --model-dir or --manifest",
        )),
    }
}

fn input_source_display(shared: &SharedSynthesizeArgs) -> &Path {
    shared
        .model_dir
        .as_deref()
        .or(shared.manifest.as_deref())
        .expect("clap requires one input source")
}

fn parse_language(value: &str) -> LanguageSelection {
    if value.trim().eq_ignore_ascii_case("auto") {
        LanguageSelection::Auto
    } else {
        LanguageSelection::Named(value.trim().to_string())
    }
}

fn resolve_stage_summary(shared: &SharedSynthesizeArgs) -> bool {
    if shared.no_profiling_stage_summary {
        false
    } else {
        shared.profiling_stage_summary
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
            "synthesize",
            "base",
            "--model-dir",
            "model-dir",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ])
        .expect("minimal args should parse");

        match args.command {
            Command::Synthesize(command) => match command.profile {
                ProfileCommand::Base(base) => {
                    assert_eq!(base.shared.model_dir, Some(PathBuf::from("model-dir")));
                    assert_eq!(base.shared.manifest, None);
                    assert_eq!(base.shared.text, "hello");
                    assert_eq!(base.shared.language, "auto");
                    assert_eq!(base.shared.output, PathBuf::from("out.wav"));
                    assert_eq!(base.shared.max_new_tokens, 256);
                }
                ProfileCommand::CustomVoice(_) => panic!("expected base command"),
            },
        }
    }

    #[test]
    fn args_parse_backend_as_enum() {
        let args = Args::try_parse_from([
            "tts_cli",
            "synthesize",
            "custom-voice",
            "--manifest",
            "package.yaml",
            "--text",
            "hello",
            "--language",
            "zh",
            "--speaker",
            "Chelsie",
            "--output",
            "out.wav",
            "--backend",
            "flex",
        ])
        .expect("backend should parse as enum");

        match args.command {
            Command::Synthesize(command) => match command.profile {
                ProfileCommand::CustomVoice(custom_voice) => {
                    assert_eq!(custom_voice.shared.backend, Some(CliBackend::Flex));
                    assert_eq!(
                        custom_voice.shared.manifest,
                        Some(PathBuf::from("package.yaml"))
                    );
                    assert_eq!(custom_voice.speaker.as_deref(), Some("Chelsie"));
                }
                ProfileCommand::Base(_) => panic!("expected custom-voice command"),
            },
        }
    }

    #[test]
    fn package_source_prefers_model_dir_and_manifest_flags() {
        let model_dir_args = SharedSynthesizeArgs {
            model_dir: Some(PathBuf::from("model-dir")),
            manifest: None,
            text: "hello".to_string(),
            language: "auto".to_string(),
            output: PathBuf::from("out.wav"),
            backend: None,
            max_new_tokens: 256,
            sampling: CliSampling::Greedy,
            profiling: false,
            profiling_per_step: false,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 8,
        };
        assert!(matches!(
            package_source(&model_dir_args).unwrap(),
            Qwen3TtsPackageSource::ModelDir(_)
        ));

        let manifest_args = SharedSynthesizeArgs {
            model_dir: None,
            manifest: Some(PathBuf::from("package.yaml")),
            ..model_dir_args
        };
        assert!(matches!(
            package_source(&manifest_args).unwrap(),
            Qwen3TtsPackageSource::ManifestPath(_)
        ));
    }

    #[test]
    fn args_reject_missing_input_source() {
        let error = Args::try_parse_from([
            "tts_cli",
            "synthesize",
            "base",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ])
        .expect_err("input source should be required");

        let message = error.to_string();
        assert!(message.contains("--model-dir") || message.contains("--manifest"));
    }
}
