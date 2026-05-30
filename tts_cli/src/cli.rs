use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{ArgAction, Args as ClapArgs, Parser, Subcommand, ValueEnum};
use tracing::info;
use tts_app::{
    BaseSynthesisInput, CustomVoiceSynthesisInput, Qwen3TtsBackend, QwenAppService, SamplingConfig,
    SharedSynthesisInput,
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
    #[arg(long)]
    pub max_new_tokens: Option<usize>,
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
    #[arg(long)]
    pub ref_audio: Option<PathBuf>,
    #[arg(long, requires = "ref_audio")]
    pub ref_text: Option<String>,
    #[arg(long, requires = "ref_audio", conflicts_with = "ref_text")]
    pub x_vector_only: bool,
}

#[derive(Debug, Clone, ClapArgs)]
pub struct CustomVoiceSynthesizeArgs {
    #[command(flatten)]
    pub shared: SharedSynthesizeArgs,
    #[arg(long)]
    pub speaker: String,
    #[arg(long)]
    pub instruct: Option<String>,
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
    let service = QwenAppService::new()?;

    match args.command {
        Command::Synthesize(command) => match command.profile {
            ProfileCommand::Base(base) => run_base_synthesis(&service, &base, total_started),
            ProfileCommand::CustomVoice(custom_voice) => {
                run_custom_voice_synthesis(&service, &custom_voice, total_started)
            }
        },
    }
}

fn run_base_synthesis(
    service: &QwenAppService,
    args: &BaseSynthesizeArgs,
    total_started: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        source = %input_source_display(&args.shared).display(),
        output = %args.shared.output.display(),
        backend = ?args.shared.backend,
        max_new_tokens = ?args.shared.max_new_tokens,
        profiling = args.shared.profiling,
        language = %args.shared.language,
        "starting tts generation"
    );

    let prepared = QwenAppService::prepare_base(BaseSynthesisInput {
        shared: to_shared_input(&args.shared),
        ref_audio: args.ref_audio.clone(),
        ref_text: args.ref_text.clone(),
        x_vector_only: args.x_vector_only,
    })?;
    let saved = service.synthesize_prepared(prepared)?;

    info!(
        wav_path = %saved.output.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        instance_id = saved.result.instance_id,
        driver_id = %saved.result.driver_id,
        synthesis_elapsed_ms = saved.result.elapsed.as_millis(),
        "saved wav"
    );
    Ok(())
}

fn run_custom_voice_synthesis(
    service: &QwenAppService,
    args: &CustomVoiceSynthesizeArgs,
    total_started: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        source = %input_source_display(&args.shared).display(),
        output = %args.shared.output.display(),
        backend = ?args.shared.backend,
        max_new_tokens = ?args.shared.max_new_tokens,
        profiling = args.shared.profiling,
        language = %args.shared.language,
        "starting tts generation"
    );

    let prepared = QwenAppService::prepare_custom_voice(CustomVoiceSynthesisInput {
        shared: to_shared_input(&args.shared),
        speaker: args.speaker.clone(),
        instruct: args.instruct.clone(),
    })?;
    let saved = service.synthesize_prepared(prepared)?;

    info!(
        wav_path = %saved.output.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        instance_id = saved.result.instance_id,
        driver_id = %saved.result.driver_id,
        synthesis_elapsed_ms = saved.result.elapsed.as_millis(),
        "saved wav"
    );
    Ok(())
}

fn to_shared_input(shared: &SharedSynthesizeArgs) -> SharedSynthesisInput {
    SharedSynthesisInput {
        model_dir: shared.model_dir.clone(),
        manifest: shared.manifest.clone(),
        text: shared.text.clone(),
        language: shared.language.clone(),
        output: shared.output.clone(),
        backend: shared.backend.map(CliBackend::to_backend),
        max_new_tokens: shared.max_new_tokens,
        sampling: shared.sampling.to_sampling(),
        profiling: shared.profiling,
        profiling_per_step: shared.profiling_per_step,
        profiling_stage_summary: shared.profiling_stage_summary,
        no_profiling_stage_summary: shared.no_profiling_stage_summary,
        profiling_log_topk: shared.profiling_log_topk,
    }
}

fn input_source_display(shared: &SharedSynthesizeArgs) -> &Path {
    shared
        .model_dir
        .as_deref()
        .or(shared.manifest.as_deref())
        .expect("clap requires one input source")
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
                    assert_eq!(base.shared.max_new_tokens, None);
                    assert_eq!(base.ref_audio, None);
                    assert_eq!(base.ref_text, None);
                    assert!(!base.x_vector_only);
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
                    assert_eq!(custom_voice.speaker, "Chelsie");
                }
                ProfileCommand::Base(_) => panic!("expected custom-voice command"),
            },
        }
    }

    #[test]
    fn args_parse_base_clone_flags() {
        let args = Args::try_parse_from([
            "tts_cli",
            "synthesize",
            "base",
            "--model-dir",
            "model-dir",
            "--text",
            "hello",
            "--ref-audio",
            "clone.wav",
            "--ref-text",
            "reference speech",
            "--output",
            "out.wav",
        ])
        .expect("base clone flags should parse");

        match args.command {
            Command::Synthesize(command) => match command.profile {
                ProfileCommand::Base(base) => {
                    assert_eq!(base.ref_audio, Some(PathBuf::from("clone.wav")));
                    assert_eq!(base.ref_text.as_deref(), Some("reference speech"));
                    assert!(!base.x_vector_only);
                }
                ProfileCommand::CustomVoice(_) => panic!("expected base command"),
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
            max_new_tokens: None,
            sampling: CliSampling::Greedy,
            profiling: false,
            profiling_per_step: false,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 8,
        };
        let model_dir_input = to_shared_input(&model_dir_args);
        assert_eq!(model_dir_input.model_dir, Some(PathBuf::from("model-dir")));
        assert_eq!(model_dir_input.manifest, None);

        let manifest_args = SharedSynthesizeArgs {
            model_dir: None,
            manifest: Some(PathBuf::from("package.yaml")),
            ..model_dir_args
        };
        let manifest_input = to_shared_input(&manifest_args);
        assert_eq!(manifest_input.model_dir, None);
        assert_eq!(manifest_input.manifest, Some(PathBuf::from("package.yaml")));
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

    #[test]
    fn args_reject_x_vector_only_with_ref_text() {
        let error = Args::try_parse_from([
            "tts_cli",
            "synthesize",
            "base",
            "--model-dir",
            "model-dir",
            "--text",
            "hello",
            "--ref-audio",
            "clone.wav",
            "--ref-text",
            "reference speech",
            "--x-vector-only",
            "--output",
            "out.wav",
        ])
        .expect_err("x-vector-only should conflict with ref-text");

        assert!(error.to_string().contains("--ref-text"));
    }

    #[test]
    fn shared_args_conversion_preserves_runtime_knobs() {
        let shared = SharedSynthesizeArgs {
            model_dir: Some(PathBuf::from("model-dir")),
            manifest: None,
            text: "hello".to_string(),
            language: "auto".to_string(),
            output: PathBuf::from("out.wav"),
            backend: Some(CliBackend::Flex),
            max_new_tokens: Some(32),
            sampling: CliSampling::Greedy,
            profiling: true,
            profiling_per_step: true,
            profiling_stage_summary: true,
            no_profiling_stage_summary: false,
            profiling_log_topk: 3,
        };

        let input = to_shared_input(&shared);
        assert_eq!(input.backend, Some(Qwen3TtsBackend::Flex));
        assert_eq!(input.max_new_tokens, Some(32));
        assert_eq!(input.sampling, SamplingConfig::greedy());
        assert!(input.profiling);
        assert!(input.profiling_per_step);
        assert_eq!(input.profiling_log_topk, 3);
    }

    #[test]
    fn custom_voice_args_keep_shell_level_fields() {
        let args = Args::try_parse_from([
            "tts_cli",
            "synthesize",
            "custom-voice",
            "--model-dir",
            "model-dir",
            "--text",
            "hello",
            "--speaker",
            "Vivian",
            "--instruct",
            "用特别愤怒的语气说",
            "--output",
            "out.wav",
        ])
        .expect("custom voice args should parse");

        match args.command {
            Command::Synthesize(command) => match command.profile {
                ProfileCommand::CustomVoice(custom_voice) => {
                    assert_eq!(custom_voice.speaker, "Vivian");
                    assert_eq!(custom_voice.instruct.as_deref(), Some("用特别愤怒的语气说"));
                }
                ProfileCommand::Base(_) => panic!("expected custom-voice command"),
            },
        }
    }
}
