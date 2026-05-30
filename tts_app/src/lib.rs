mod backend;

use std::path::PathBuf;

use thiserror::Error;
use tts_core::{DriverRegistry, ModelManager, SynthesisResult};
use tts_qwen3_tts::{
    register_driver, BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio,
    CustomVoiceRequest, LanguageSelection, Qwen3TtsEngineConfig, Qwen3TtsHandleExt,
    Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

pub use self::backend::{available_backends, resolve_backend};
pub use tts_qwen3_tts::{Qwen3TtsBackend, SamplingConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct SharedSynthesisInput {
    pub model_dir: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
    pub text: String,
    pub language: String,
    pub output: PathBuf,
    pub backend: Option<Qwen3TtsBackend>,
    pub max_new_tokens: Option<usize>,
    pub sampling: SamplingConfig,
    pub profiling: bool,
    pub profiling_per_step: bool,
    pub profiling_stage_summary: bool,
    pub no_profiling_stage_summary: bool,
    pub profiling_log_topk: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaseSynthesisInput {
    pub shared: SharedSynthesisInput,
    pub ref_audio: Option<PathBuf>,
    pub ref_text: Option<String>,
    pub x_vector_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomVoiceSynthesisInput {
    pub shared: SharedSynthesisInput,
    pub speaker: String,
    pub instruct: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSynthesis {
    pub package_source: Qwen3TtsPackageSource,
    pub backend: Qwen3TtsBackend,
    pub request: QwenRequest,
    pub output: PathBuf,
    pub profiling: Qwen3TtsProfilingConfig,
    pub run_options: Qwen3TtsRunOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavedSynthesis {
    pub output: PathBuf,
    pub result: SynthesisResult,
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Diagnostic(#[from] tts_error::DiagnosticError),
    #[error(transparent)]
    Model(#[from] tts_qwen3_tts::Qwen3TtsError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Clone)]
pub struct QwenAppService {
    manager: ModelManager,
}

impl QwenAppService {
    pub fn new() -> Result<Self, AppError> {
        let mut registry = DriverRegistry::new();
        register_driver(&mut registry)?;
        Ok(Self {
            manager: ModelManager::new(registry),
        })
    }
}

impl QwenAppService {
    pub fn build_base_request(input: BaseSynthesisInput) -> Result<QwenRequest, AppError> {
        let voice_clone = match (&input.ref_audio, input.x_vector_only) {
            (None, _) => {
                if input.ref_text.is_some() {
                    return Err(tts_error::DiagnosticError::invalid_argument(
                        "app.ref_text_requires_audio",
                        "`--ref-text` requires `--ref-audio`",
                    )
                    .into());
                }
                None
            }
            (Some(_), false) if input.ref_text.is_none() => {
                return Err(tts_error::DiagnosticError::invalid_argument(
                    "app.ref_text_required",
                    "`--ref-text` is required when `--ref-audio` is used without `--x-vector-only`",
                )
                .into());
            }
            (Some(path), x_vector_only) => Some(BaseVoiceCloneConditioning::ReferenceAudio(
                BaseVoiceCloneReferenceAudio {
                    path: path.clone(),
                    transcript: input.ref_text.clone(),
                    x_vector_only,
                },
            )),
        };

        Ok(QwenRequest::Base(BaseRequest {
            text: input.shared.text,
            language: parse_language(&input.shared.language),
            voice_clone,
        }))
    }

    pub fn prepare_base(input: BaseSynthesisInput) -> Result<PreparedSynthesis, AppError> {
        let stage_summary = resolve_stage_summary(&input.shared);
        let output = input.shared.output.clone();
        let backend = resolve_backend(input.shared.backend)?;
        Ok(PreparedSynthesis {
            package_source: package_source(&input.shared)?,
            backend,
            request: Self::build_base_request(input.clone())?,
            output,
            profiling: Qwen3TtsProfilingConfig {
                enabled: input.shared.profiling,
                per_step: input.shared.profiling_per_step,
                stage_summary,
                log_topk: input.shared.profiling_log_topk,
            },
            run_options: Qwen3TtsRunOptions {
                max_new_tokens: input.shared.max_new_tokens,
                sampling: input.shared.sampling,
            },
        })
    }

    pub fn prepare_custom_voice(
        input: CustomVoiceSynthesisInput,
    ) -> Result<PreparedSynthesis, AppError> {
        let stage_summary = resolve_stage_summary(&input.shared);
        let output = input.shared.output.clone();
        let backend = resolve_backend(input.shared.backend)?;
        Ok(PreparedSynthesis {
            package_source: package_source(&input.shared)?,
            backend,
            request: QwenRequest::CustomVoice(CustomVoiceRequest {
                text: input.shared.text,
                language: parse_language(&input.shared.language),
                speaker: Some(input.speaker),
                instruct: input.instruct,
            }),
            output,
            profiling: Qwen3TtsProfilingConfig {
                enabled: input.shared.profiling,
                per_step: input.shared.profiling_per_step,
                stage_summary,
                log_topk: input.shared.profiling_log_topk,
            },
            run_options: Qwen3TtsRunOptions {
                max_new_tokens: input.shared.max_new_tokens,
                sampling: input.shared.sampling,
            },
        })
    }

    pub fn synthesize_prepared(
        &self,
        prepared: PreparedSynthesis,
    ) -> Result<SavedSynthesis, AppError> {
        if let Some(parent) = prepared
            .output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }

        let handle = self.manager.load(
            tts_qwen3_tts::QWEN3_TTS_DRIVER_ID,
            Qwen3TtsEngineConfig {
                package: prepared.package_source.clone(),
                backend: prepared.backend,
                profiling: prepared.profiling.clone(),
            },
        )?;

        let result = handle.synthesize_qwen(prepared.request, prepared.run_options)?;
        result.audio.save_wav(&prepared.output)?;
        handle.close()?;
        let _ = self.manager.remove(handle.instance_id())?;

        Ok(SavedSynthesis {
            output: prepared.output,
            result,
        })
    }
}

fn package_source(shared: &SharedSynthesisInput) -> Result<Qwen3TtsPackageSource, AppError> {
    match (&shared.model_dir, &shared.manifest) {
        (Some(path), None) => Ok(Qwen3TtsPackageSource::ModelDir(path.clone())),
        (None, Some(path)) => Ok(Qwen3TtsPackageSource::ManifestPath(path.clone())),
        (None, None) => Err(tts_error::DiagnosticError::invalid_argument(
            "app.input_source_required",
            "pass either --model-dir or --manifest",
        )
        .into()),
        (Some(_), Some(_)) => Err(tts_error::DiagnosticError::invalid_argument(
            "app.input_source_conflict",
            "pass only one of --model-dir or --manifest",
        )
        .into()),
    }
}

fn parse_language(value: &str) -> LanguageSelection {
    if value.trim().eq_ignore_ascii_case("auto") {
        LanguageSelection::Auto
    } else {
        LanguageSelection::Named(value.trim().to_string())
    }
}

fn resolve_stage_summary(shared: &SharedSynthesisInput) -> bool {
    if shared.no_profiling_stage_summary {
        false
    } else {
        shared.profiling_stage_summary
    }
}
