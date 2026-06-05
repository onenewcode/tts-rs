pub(crate) mod load_adapter;
pub(crate) mod package;
pub(crate) mod runtime;

use burn::tensor::{DType, FloatDType};

use crate::capabilities::project_capabilities;
use crate::execution::Qwen3LoadedModelInstance;
use crate::execution::Qwen3TtsLoadedModel;
use crate::execution::compiler::Qwen3TtsRequestCompiler;
use crate::loading::package::{Qwen3TtsPackage, Qwen3TtsPackageSource};
use crate::loading::runtime::{RuntimeBackend, build_runtime};
use crate::model::speaker::config::SpeakerConfigEnvelope;
use crate::{Qwen3TtsEngineConfig, Qwen3TtsLoadError};

#[derive(Debug, Clone)]
pub(crate) struct ResolvedPackage {
    pub(crate) package: Qwen3TtsPackage,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
    pub(crate) has_speaker_encoder: bool,
}

pub(crate) fn load_instance(
    config: &Qwen3TtsEngineConfig,
) -> Result<Qwen3LoadedModelInstance, Qwen3TtsLoadError> {
    let resolved = resolve_package(&config.package)?;
    validate_runtime_dtype(config.talker_dtype)?;
    validate_runtime_dtype(config.codec_dtype)?;
    let device = Default::default();
    let runtime = build_runtime::<RuntimeBackend>(
        &resolved,
        config.talker_dtype.map(DType::from),
        config.codec_dtype.map(DType::from),
        &device,
    )?;
    tracing::info!(
        talker_dtype = %config
            .talker_dtype
            .map(DType::from)
            .map(|dtype| dtype.name())
            .unwrap_or("original"),
        codec_dtype = %config
            .codec_dtype
            .map(DType::from)
            .map(|dtype| dtype.name())
            .unwrap_or("original"),
        "assembled qwen3 runtime"
    );
    let model = Qwen3TtsLoadedModel::new(runtime);
    let capabilities = project_capabilities(&resolved);
    Ok(Qwen3LoadedModelInstance::new(
        model,
        resolved.package,
        config.profiling.clone(),
        capabilities,
    ))
}

fn resolve_package(source: &Qwen3TtsPackageSource) -> Result<ResolvedPackage, Qwen3TtsLoadError> {
    let package = Qwen3TtsPackage::load(source)?;
    let compiler = Qwen3TtsRequestCompiler::load(&package)?;
    let raw = std::fs::read_to_string(&package.talker_config_path).map_err(|source| {
        Qwen3TtsLoadError::CompilerConfigIo {
            path: package.talker_config_path.clone(),
            source,
        }
    })?;
    let envelope: SpeakerConfigEnvelope =
        serde_json::from_str(&raw).map_err(|source| Qwen3TtsLoadError::CompilerConfigParse {
            path: package.talker_config_path.clone(),
            source,
        })?;
    Ok(ResolvedPackage {
        package,
        compiler,
        has_speaker_encoder: envelope.speaker_encoder_config.is_some(),
    })
}

fn validate_runtime_dtype(dtype: Option<FloatDType>) -> Result<(), Qwen3TtsLoadError> {
    if let Some(dtype) = dtype
        && !matches!(dtype, FloatDType::F16 | FloatDType::F32 | FloatDType::BF16)
    {
        return Err(Qwen3TtsLoadError::UnsupportedDType {
            requested: DType::from(dtype).name().to_string(),
        });
    }

    Ok(())
}
