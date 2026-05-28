use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;
use tracing::info;
use tts_core::{
    ComputeBackend, ModelRegistry, SynthesisOptions, SynthesisRequest, TtsService, save_pcm_wav,
};
use tts_qwen::register_qwen_family_model;

#[derive(Debug, Parser)]
#[command(name = "tts_cli")]
pub struct Args {
    #[arg(long)]
    pub models_config: PathBuf,
    #[arg(long)]
    pub model_id: String,
    #[arg(long)]
    pub text: String,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long)]
    pub speaker: Option<String>,
    #[arg(long, value_enum)]
    pub backend: Option<CliBackend>,
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,
    #[arg(long)]
    pub max_new_tokens: Option<usize>,
    #[arg(long)]
    pub chunk_steps: Option<usize>,
    #[arg(long)]
    pub stream: bool,
    #[arg(long)]
    pub profiling: bool,
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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
    fn to_backend(self) -> ComputeBackend {
        match self {
            Self::Flex => ComputeBackend::Flex,
            Self::Wgpu => ComputeBackend::Wgpu,
            Self::Cuda => ComputeBackend::Cuda,
            Self::Rocm => ComputeBackend::Rocm,
            Self::Metal => ComputeBackend::Metal,
            Self::Vulkan => ComputeBackend::Vulkan,
            Self::Webgpu => ComputeBackend::WebGpu,
        }
    }
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

#[derive(Debug, Deserialize)]
struct ModelsConfig {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    family: String,
    variant: String,
    paths: ModelPaths,
    defaults: Option<ModelDefaults>,
}

#[derive(Debug, Deserialize)]
struct ModelPaths {
    model_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelDefaults {
    max_new_tokens: Option<usize>,
    chunk_steps: Option<usize>,
    stream: Option<bool>,
    profiling: Option<bool>,
    backend: Option<CliBackend>,
}

pub fn run_from_args() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    init_logging(args.log_level);
    let total_started = Instant::now();
    std::fs::create_dir_all(&args.output_dir)?;

    let config = load_models_config(&args.models_config)?;
    let (service, defaults_by_model) = build_service(config)?;
    let defaults = defaults_by_model
        .get(&args.model_id)
        .cloned()
        .ok_or_else(|| format!("model_id `{}` is not defined in config", args.model_id))?;

    let options = merge_options(&args, defaults);
    info!(
        models_config = %args.models_config.display(),
        model_id = %args.model_id,
        output_dir = %args.output_dir.display(),
        backend = ?options.backend,
        max_new_tokens = options.max_new_tokens,
        chunk_steps = options.chunk_steps,
        stream = options.stream,
        profiling = options.profiling,
        language = args.language.as_deref().unwrap_or("Auto"),
        speaker = args.speaker.as_deref().unwrap_or(""),
        "starting tts generation"
    );

    let request = SynthesisRequest {
        text: args.text.clone(),
        language: args.language.clone(),
        speaker: args.speaker.clone(),
    };
    let result = service
        .synthesize(&args.model_id, &request, &options)
        .map_err(|error| format!("tts inference failed: {error}"))?;

    let wav_path = args.output_dir.join("0000.wav");
    save_pcm_wav(&result.waveform_pcm, &wav_path, result.sample_rate)
        .map_err(|error| format!("failed to save wav: {error}"))?;
    info!(
        wav_path = %wav_path.display(),
        total_elapsed_ms = total_started.elapsed().as_millis(),
        "saved wav"
    );
    Ok(())
}

fn load_models_config(path: &PathBuf) -> Result<ModelsConfig, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let config = serde_yaml::from_str::<ModelsConfig>(&raw)?;
    Ok(config)
}

fn build_service(
    config: ModelsConfig,
) -> Result<(TtsService, HashMap<String, ModelDefaults>), Box<dyn std::error::Error>> {
    if config.models.is_empty() {
        return Err("models config must contain at least one model entry".into());
    }

    let mut registry = ModelRegistry::new();
    let mut defaults = HashMap::new();

    for model in config.models {
        match model.family.as_str() {
            "qwen" => {
                if !register_qwen_family_model(
                    &mut registry,
                    model.id.clone(),
                    model.paths.model_dir,
                    model.variant,
                ) {
                    return Err(format!("duplicate model id `{}` in config", model.id).into());
                }
                defaults.insert(model.id, model.defaults.unwrap_or_default());
            }
            other => {
                return Err(
                    format!("unsupported model family `{other}`; supported values: qwen").into(),
                );
            }
        }
    }

    Ok((TtsService::new(registry), defaults))
}

fn merge_options(args: &Args, defaults: ModelDefaults) -> SynthesisOptions {
    let base = SynthesisOptions::default();
    SynthesisOptions {
        max_new_tokens: args
            .max_new_tokens
            .or(defaults.max_new_tokens)
            .unwrap_or(base.max_new_tokens),
        chunk_steps: args
            .chunk_steps
            .or(defaults.chunk_steps)
            .unwrap_or(base.chunk_steps),
        sampling: base.sampling,
        stream: if args.stream {
            true
        } else {
            defaults.stream.unwrap_or(base.stream)
        },
        profiling: if args.profiling {
            true
        } else {
            defaults.profiling.unwrap_or(base.profiling)
        },
        backend: args
            .backend
            .map(CliBackend::to_backend)
            .or_else(|| defaults.backend.map(CliBackend::to_backend))
            .or(base.backend),
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
            "--models-config",
            "models.yaml",
            "--model-id",
            "qwen-default",
            "--text",
            "hello",
        ])
        .expect("minimal args should parse");

        assert_eq!(args.models_config, PathBuf::from("models.yaml"));
        assert_eq!(args.model_id, "qwen-default");
        assert_eq!(args.text, "hello");
        assert_eq!(args.output_dir, PathBuf::from("output"));
        assert_eq!(args.max_new_tokens, None);
        assert_eq!(args.chunk_steps, None);
        assert!(!args.stream);
        assert!(!args.profiling);
        assert_eq!(args.log_level, LogLevel::Info);
    }

    #[test]
    fn args_parse_backend_as_enum() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--models-config",
            "models.yaml",
            "--model-id",
            "qwen-default",
            "--text",
            "hello",
            "--backend",
            "flex",
        ])
        .expect("backend arg should parse");

        assert_eq!(args.backend, Some(CliBackend::Flex));
    }
}
