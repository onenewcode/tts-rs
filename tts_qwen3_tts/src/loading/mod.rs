pub(crate) mod load_adapter;
pub(crate) mod package;
pub(crate) mod runtime;

use burn::tensor::backend::{Backend, DeviceError, get_device_settings, set_default_float_dtype};
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
    let requested_dtype = config.dtype.unwrap_or(FloatDType::BF16);
    if !matches!(requested_dtype, FloatDType::F32 | FloatDType::BF16) {
        return Err(Qwen3TtsLoadError::UnsupportedDType {
            requested: DType::from(requested_dtype).name().to_string(),
        });
    }
    let device = Default::default();
    let tensor_dtype = DType::from(requested_dtype);
    initialize_device_dtype::<RuntimeBackend>(&device, requested_dtype)?;
    let runtime = build_runtime::<RuntimeBackend>(&resolved, tensor_dtype, &device)?;
    tracing::info!(dtype = %tensor_dtype.name(), "assembled qwen3 runtime");
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

pub(crate) fn initialize_device_dtype<B: Backend>(
    device: &B::Device,
    model_dtype: FloatDType,
) -> Result<(), Qwen3TtsLoadError> {
    match set_default_float_dtype::<B>(device, model_dtype) {
        Ok(()) => Ok(()),
        Err(DeviceError::AlreadyInitialized { .. })
            if get_device_settings::<B>(device).float_dtype == model_dtype =>
        {
            Ok(())
        }
        Err(source) => Err(Qwen3TtsLoadError::RuntimeDType {
            requested: DType::from(model_dtype).name().to_string(),
            message: source.to_string(),
        }),
    }
}
