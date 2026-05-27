use std::fmt::{self, Display};
use std::path::Path;
use std::str::FromStr;

use crate::{
    CustomVoiceRequest, EngineConfig, FinishedInference, ProfilingConfig, QwenTtsError,
    QwenTtsInferenceError, SamplingConfig, SessionConfig, StreamingMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Flex,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    WebGpu,
}

impl BackendKind {
    pub const ALL_LABELS: [&str; 7] = ["flex", "wgpu", "cuda", "rocm", "metal", "vulkan", "webgpu"];

    pub fn label(self) -> &'static str {
        match self {
            Self::Flex => "flex",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::WebGpu => "webgpu",
        }
    }

    fn is_compiled(self) -> bool {
        match self {
            Self::Flex => cfg!(feature = "flex"),
            Self::Wgpu => cfg!(feature = "wgpu"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Rocm => cfg!(feature = "rocm"),
            Self::Metal => cfg!(feature = "metal"),
            Self::Vulkan => cfg!(feature = "vulkan"),
            Self::WebGpu => cfg!(feature = "webgpu"),
        }
    }
}

impl Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for BackendKind {
    type Err = QwenTtsInferenceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "flex" => Ok(Self::Flex),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            "rocm" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "webgpu" => Ok(Self::WebGpu),
            other => Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "unsupported backend `{other}`; expected one of: {}",
                    Self::ALL_LABELS.join(", ")
                ),
            }),
        }
    }
}

pub fn available_backends() -> Vec<BackendKind> {
    #[allow(unused_mut)]
    let mut backends = Vec::new();

    #[cfg(feature = "flex")]
    backends.push(BackendKind::Flex);
    #[cfg(feature = "wgpu")]
    backends.push(BackendKind::Wgpu);
    #[cfg(feature = "cuda")]
    backends.push(BackendKind::Cuda);
    #[cfg(feature = "rocm")]
    backends.push(BackendKind::Rocm);
    #[cfg(feature = "metal")]
    backends.push(BackendKind::Metal);
    #[cfg(feature = "vulkan")]
    backends.push(BackendKind::Vulkan);
    #[cfg(feature = "webgpu")]
    backends.push(BackendKind::WebGpu);

    backends
}

pub fn resolve_backend(
    selected: Option<BackendKind>,
) -> Result<BackendKind, QwenTtsInferenceError> {
    select_backend(selected, &available_backends())
}

pub fn run_with_backend(
    selected: BackendKind,
    model_dir: impl AsRef<Path>,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
) -> Result<FinishedInference, QwenTtsError> {
    let backend = resolve_backend(Some(selected))?;
    let model_dir = model_dir.as_ref();

    match backend {
        BackendKind::Flex => {
            #[cfg(feature = "flex")]
            {
                return run_on_default_device::<FlexBackend>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                );
            }
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                return run_on_wgpu_device::<WgpuBackend, _>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                    |_| {},
                );
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                return run_on_default_device::<CudaBackend>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                );
            }
        }
        BackendKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                return run_on_default_device::<RocmBackend>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                );
            }
        }
        BackendKind::Metal => {
            #[cfg(feature = "metal")]
            {
                return run_on_wgpu_device::<MetalBackend, _>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                            device,
                            Default::default(),
                        );
                    },
                );
            }
        }
        BackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                return run_on_wgpu_device::<VulkanBackend, _>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                            device,
                            Default::default(),
                        );
                    },
                );
            }
        }
        BackendKind::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                return run_on_wgpu_device::<WebGpuBackend, _>(
                    model_dir,
                    request,
                    engine_config,
                    session_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::WebGpu>(
                            device,
                            Default::default(),
                        );
                    },
                );
            }
        }
    }

    let _ = (model_dir, request, engine_config, session_config);
    Err(QwenTtsInferenceError::InvalidInput {
        message: format!(
            "backend `{backend}` is not compiled in; available backends: {}",
            format_available_backends(&available_backends())
        ),
    }
    .into())
}

pub fn default_engine_config(chunk_steps: usize, profiling: bool) -> EngineConfig {
    EngineConfig {
        codec_chunk_steps: chunk_steps,
        profiling: ProfilingConfig {
            enabled: profiling,
            per_step: profiling,
            stage_summary: true,
            log_topk: 8,
        },
        ..EngineConfig::default()
    }
}

pub fn default_session_config(max_new_tokens: usize, stream: bool) -> SessionConfig {
    SessionConfig {
        max_new_tokens,
        sampling: SamplingConfig::greedy(),
        streaming: if stream {
            StreamingMode::AudioChunks
        } else {
            StreamingMode::Full
        },
    }
}

fn select_backend(
    selected: Option<BackendKind>,
    available: &[BackendKind],
) -> Result<BackendKind, QwenTtsInferenceError> {
    match selected {
        Some(backend) if backend.is_compiled() => Ok(backend),
        Some(backend) => Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "backend `{backend}` is not compiled in; available backends: {}",
                format_available_backends(available)
            ),
        }),
        None if available.is_empty() => Err(QwenTtsInferenceError::InvalidInput {
            message: "no runtime backend is compiled in; enable one of: flex, wgpu, cuda, rocm, metal, vulkan, webgpu"
                .to_string(),
        }),
        None if available.len() == 1 => Ok(available[0]),
        None => Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "multiple backends are compiled in; pass --backend one of: {}",
                format_available_backends(available)
            ),
        }),
    }
}

fn format_available_backends(backends: &[BackendKind]) -> String {
    if backends.is_empty() {
        "none".to_string()
    } else {
        backends
            .iter()
            .map(|backend| backend.label())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(any(
    feature = "flex",
    feature = "cuda",
    feature = "rocm",
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
#[allow(dead_code)]
fn run_on_default_device<B>(
    model_dir: &Path,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
) -> Result<FinishedInference, QwenTtsError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    run_on_device::<B>(model_dir, &device, request, engine_config, session_config)
}

#[cfg(any(
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
fn run_on_wgpu_device<B, F>(
    model_dir: &Path,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
    init: F,
) -> Result<FinishedInference, QwenTtsError>
where
    B: burn::tensor::backend::Backend<Device = burn::backend::wgpu::WgpuDevice>,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = Default::default();
    init(&device);
    run_on_device::<B>(model_dir, &device, request, engine_config, session_config)
}

#[cfg(any(
    feature = "flex",
    feature = "cuda",
    feature = "rocm",
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
fn run_on_device<B>(
    model_dir: &Path,
    device: &B::Device,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
) -> Result<FinishedInference, QwenTtsError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    let mut engine = crate::QwenTtsEngine::<B>::load(model_dir, device, engine_config)?;
    let handle = engine.start_session(request, session_config)?;

    if matches!(engine.step(handle)?, crate::StepOutcome::Finished) {
        return engine.finish_session(handle);
    }

    loop {
        let _ = engine.drain_events(handle)?;
        if matches!(engine.step(handle)?, crate::StepOutcome::Finished) {
            return engine.finish_session(handle);
        }
    }
}

#[cfg(feature = "flex")]
type FlexBackend = burn::backend::Flex;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn::backend::Wgpu;

#[cfg(feature = "cuda")]
type CudaBackend = burn::backend::Cuda;

#[cfg(feature = "rocm")]
type RocmBackend = burn::backend::Rocm;

#[cfg(feature = "metal")]
type MetalBackend = burn::backend::Metal;

#[cfg(feature = "vulkan")]
type VulkanBackend = burn::backend::Vulkan;

#[cfg(feature = "webgpu")]
type WebGpuBackend = burn::backend::WebGpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_backend_rejects_missing_compiled_backends() {
        let error = select_backend(None, &[]).unwrap_err().to_string();
        assert!(error.contains("no runtime backend is compiled in"));
    }

    #[test]
    fn select_backend_uses_the_only_compiled_backend() {
        let selected = select_backend(None, &[BackendKind::Flex]).unwrap();
        assert_eq!(selected, BackendKind::Flex);
    }

    #[test]
    fn select_backend_requires_explicit_choice_with_multiple_backends() {
        let error = select_backend(None, &[BackendKind::Flex, BackendKind::Cuda])
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple backends are compiled in"));
    }
}
