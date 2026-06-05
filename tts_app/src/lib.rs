pub use tts_qwen3_tts::FloatDType;

use std::path::PathBuf;

use thiserror::Error;
use tts_infer::{DriverRegistry, ModelManager, SynthesisResult};
use tts_qwen3_tts::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    LanguageSelection, Qwen3TtsDriver, Qwen3TtsEngineConfig, Qwen3TtsHandleExt,
    Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
};

pub use tts_qwen3_tts::SamplingConfig;
pub use tts_qwen3_tts::SamplingOverride;

#[derive(Debug, Clone, PartialEq)]
pub struct SharedSynthesisInput {
    pub model_dir: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
    pub text: String,
    pub language: String,
    pub output: PathBuf,
    pub max_new_tokens: Option<usize>,
    pub talker_sampling: Option<SamplingOverride>,
    pub code_predictor_sampling: Option<SamplingOverride>,
    pub talker_dtype: Option<FloatDType>,
    pub codec_dtype: Option<FloatDType>,
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
    pub request: QwenRequest,
    pub output: PathBuf,
    pub profiling: Qwen3TtsProfilingConfig,
    pub run_options: Qwen3TtsRunOptions,
    pub talker_dtype: Option<FloatDType>,
    pub codec_dtype: Option<FloatDType>,
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
        registry.register(Qwen3TtsDriver)?;
        Ok(Self {
            manager: ModelManager::new(registry),
        })
    }
}

impl QwenAppService {
    pub fn build_base_request(input: BaseSynthesisInput) -> Result<QwenRequest, AppError> {
        let BaseSynthesisInput {
            shared,
            ref_audio,
            ref_text,
            x_vector_only,
        } = input;
        let voice_clone = base_voice_clone_conditioning(ref_audio, ref_text, x_vector_only)?;

        Ok(QwenRequest::Base(BaseRequest {
            text: shared.text,
            language: parse_language(&shared.language),
            voice_clone,
        }))
    }

    pub fn prepare_base(input: BaseSynthesisInput) -> Result<PreparedSynthesis, AppError> {
        let BaseSynthesisInput {
            shared,
            ref_audio,
            ref_text,
            x_vector_only,
        } = input;
        if shared.max_new_tokens == Some(0) {
            return Err(tts_error::DiagnosticError::invalid_argument(
                "app.max_new_tokens_invalid",
                "`max_new_tokens` must be greater than zero",
            )
            .into());
        }
        let stage_summary = resolve_stage_summary(&shared);
        let package_source = package_source(&shared)?;
        let voice_clone = base_voice_clone_conditioning(ref_audio, ref_text, x_vector_only)?;
        let SharedSynthesisInput {
            text,
            language,
            output,
            max_new_tokens,
            talker_sampling,
            code_predictor_sampling,
            talker_dtype,
            codec_dtype,
            profiling,
            profiling_per_step,
            profiling_log_topk,
            ..
        } = shared;

        let request = QwenRequest::Base(BaseRequest {
            text,
            language: parse_language(&language),
            voice_clone,
        });
        let profiling = Qwen3TtsProfilingConfig {
            enabled: profiling,
            per_step: profiling_per_step,
            stage_summary,
            log_topk: profiling_log_topk,
        };
        let run_options = Qwen3TtsRunOptions {
            max_new_tokens,
            talker_sampling,
            code_predictor_sampling,
        };
        Ok(PreparedSynthesis {
            package_source,
            request,
            output,
            profiling,
            run_options,
            talker_dtype,
            codec_dtype,
        })
    }

    pub fn prepare_custom_voice(
        input: CustomVoiceSynthesisInput,
    ) -> Result<PreparedSynthesis, AppError> {
        if input.shared.max_new_tokens == Some(0) {
            return Err(tts_error::DiagnosticError::invalid_argument(
                "app.max_new_tokens_invalid",
                "`max_new_tokens` must be greater than zero",
            )
            .into());
        }
        let stage_summary = resolve_stage_summary(&input.shared);
        let output = input.shared.output.clone();
        Ok(PreparedSynthesis {
            package_source: package_source(&input.shared)?,
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
                talker_sampling: input.shared.talker_sampling,
                code_predictor_sampling: input.shared.code_predictor_sampling,
            },
            talker_dtype: input.shared.talker_dtype,
            codec_dtype: input.shared.codec_dtype,
        })
    }

    pub fn synthesize_prepared(
        &self,
        prepared: PreparedSynthesis,
    ) -> Result<SavedSynthesis, AppError> {
        let PreparedSynthesis {
            package_source,
            request,
            output,
            profiling,
            run_options,
            talker_dtype,
            codec_dtype,
        } = prepared;

        if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        let handle = self.manager.load(
            tts_qwen3_tts::QWEN3_TTS_DRIVER_ID,
            Qwen3TtsEngineConfig {
                package: package_source,
                profiling,
                talker_dtype,
                codec_dtype,
            },
        )?;

        let result = handle.synthesize_qwen(request, run_options)?;
        result.audio.save_wav(&output)?;
        handle.close()?;
        let _ = self.manager.remove(handle.instance_id())?;

        Ok(SavedSynthesis { output, result })
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

fn base_voice_clone_conditioning(
    ref_audio: Option<PathBuf>,
    ref_text: Option<String>,
    x_vector_only: bool,
) -> Result<Option<BaseVoiceCloneConditioning>, AppError> {
    match (ref_audio, ref_text, x_vector_only) {
        (None, Some(_), _) => Err(tts_error::DiagnosticError::invalid_argument(
            "app.ref_text_requires_audio",
            "`--ref-text` requires `--ref-audio`",
        )
        .into()),
        (None, None, _) => Ok(None),
        (Some(_), None, false) => Err(tts_error::DiagnosticError::invalid_argument(
            "app.ref_text_required",
            "`--ref-text` is required when `--ref-audio` is used without `--x-vector-only`",
        )
        .into()),
        (Some(path), transcript, x_vector_only) => Ok(Some(
            BaseVoiceCloneConditioning::ReferenceAudio(BaseVoiceCloneReferenceAudio {
                path,
                transcript,
                x_vector_only,
            }),
        )),
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
