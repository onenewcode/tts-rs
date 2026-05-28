use std::path::{Path, PathBuf};
use std::sync::Arc;

use tts_core::{
    ComputeBackend, ModelCapabilities, ModelRegistry, ModelStep, SynthesisOptions, SynthesisResult,
    TtsCoreError, TtsModelExecutor, TtsModelRun,
};

use crate::arch::engine::assembly::{
    EngineArtifact, FinishedInference, QwenRun, QwenRunConfig, QwenRunStep,
};
use crate::profile::QwenRequest;
use crate::releases::{QwenReleaseManifest, parse_release_manifest};
use crate::runtime::types::{EngineConfig, RunConfig, RunStep};
use crate::{BackendKind, ProfilingConfig, resolve_backend};

pub(crate) struct QwenFamilyExecutor {
    model_dir: PathBuf,
    release: &'static QwenReleaseManifest,
}

pub(crate) fn register_qwen_family_model_impl(
    registry: &mut ModelRegistry,
    model_id: impl Into<String>,
    model_dir: impl AsRef<Path>,
    variant: impl AsRef<str>,
) -> Result<bool, TtsCoreError> {
    let release = parse_release_manifest(variant.as_ref())?;
    Ok(registry
        .register(
            model_id.into(),
            Arc::new(QwenFamilyExecutor::new(model_dir, release)),
        )
        .is_none())
}

impl QwenFamilyExecutor {
    fn new(model_dir: impl AsRef<Path>, release: &'static QwenReleaseManifest) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            release,
        }
    }

    fn resolve_backend(
        &self,
        backend: Option<&ComputeBackend>,
    ) -> Result<BackendKind, TtsCoreError> {
        resolve_backend(backend.map(map_backend)).map_err(to_executor_error)
    }

    fn build_engine_config(options: &SynthesisOptions) -> EngineConfig {
        EngineConfig {
            profiling: ProfilingConfig {
                enabled: options.profiling,
                per_step: options.profiling,
                stage_summary: true,
                log_topk: 8,
            },
        }
    }

    fn build_run_config(options: &SynthesisOptions) -> RunConfig {
        RunConfig {
            max_new_tokens: options.max_new_tokens,
            sampling: options.sampling.clone(),
        }
    }
}

impl TtsModelExecutor for QwenFamilyExecutor {
    fn family(&self) -> &'static str {
        "qwen"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_streaming: true,
            supports_custom_voice: true,
        }
    }

    fn start_run(
        &self,
        request: &tts_core::SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsModelRun>, TtsCoreError> {
        let backend = self.resolve_backend(options.backend.as_ref())?;
        let request = QwenRequest::from(request);
        let engine_config = Self::build_engine_config(options);
        let run_config = Self::build_run_config(options);

        Ok(Box::new(build_release_backend_run(
            &self.model_dir,
            self.release,
            backend,
            request,
            engine_config,
            run_config,
        )?))
    }
}

struct BackendRun<B>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    engine: EngineArtifact<B>,
    device: B::Device,
    run: QwenRun<B>,
}

impl<B> BackendRun<B>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    fn advance(&mut self) -> Result<ModelStep, TtsCoreError> {
        let step = self
            .engine
            .step_run(&mut self.run)
            .map_err(to_executor_error)?;
        Ok(map_step(step))
    }

    fn decode_audio(&self) -> Result<SynthesisResult, TtsCoreError> {
        self.engine
            .snapshot_audio(&self.run, &self.device)
            .map_err(to_executor_error)
            .map(map_result)
    }

    fn finish(self) -> Result<SynthesisResult, TtsCoreError> {
        self.engine
            .finish_run(self.run, &self.device)
            .map_err(to_executor_error)
            .map(map_result)
    }
}

enum QwenBackendRun {
    #[cfg(feature = "flex")]
    Flex(BackendRun<burn::backend::Flex>),
    #[cfg(feature = "wgpu")]
    Wgpu(BackendRun<burn::backend::Wgpu>),
    #[cfg(feature = "cuda")]
    Cuda(BackendRun<burn::backend::Cuda>),
    #[cfg(feature = "rocm")]
    Rocm(BackendRun<burn::backend::Rocm>),
    #[cfg(feature = "metal")]
    Metal(BackendRun<burn::backend::Metal>),
    #[cfg(feature = "vulkan")]
    Vulkan(BackendRun<burn::backend::Vulkan>),
    #[cfg(feature = "webgpu")]
    WebGpu(BackendRun<burn::backend::WebGpu>),
}

impl TtsModelRun for QwenBackendRun {
    fn advance(&mut self) -> Result<ModelStep, TtsCoreError> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(run) => run.advance(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(run) => run.advance(),
            #[cfg(feature = "cuda")]
            Self::Cuda(run) => run.advance(),
            #[cfg(feature = "rocm")]
            Self::Rocm(run) => run.advance(),
            #[cfg(feature = "metal")]
            Self::Metal(run) => run.advance(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(run) => run.advance(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(run) => run.advance(),
        }
    }

    fn decode_audio(&self) -> Result<SynthesisResult, TtsCoreError> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(run) => run.decode_audio(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(run) => run.decode_audio(),
            #[cfg(feature = "cuda")]
            Self::Cuda(run) => run.decode_audio(),
            #[cfg(feature = "rocm")]
            Self::Rocm(run) => run.decode_audio(),
            #[cfg(feature = "metal")]
            Self::Metal(run) => run.decode_audio(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(run) => run.decode_audio(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(run) => run.decode_audio(),
        }
    }

    fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError> {
        match *self {
            #[cfg(feature = "flex")]
            Self::Flex(run) => run.finish(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(run) => run.finish(),
            #[cfg(feature = "cuda")]
            Self::Cuda(run) => run.finish(),
            #[cfg(feature = "rocm")]
            Self::Rocm(run) => run.finish(),
            #[cfg(feature = "metal")]
            Self::Metal(run) => run.finish(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(run) => run.finish(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(run) => run.finish(),
        }
    }
}

fn build_release_backend_run(
    model_dir: &Path,
    release: &'static QwenReleaseManifest,
    backend: BackendKind,
    request: QwenRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<QwenBackendRun, TtsCoreError> {
    match backend {
        BackendKind::Flex => {
            #[cfg(feature = "flex")]
            {
                Ok(QwenBackendRun::Flex(build_default_run::<
                    burn::backend::Flex,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                )?))
            }
            #[cfg(not(feature = "flex"))]
            {
                Err(unavailable_backend_error("flex"))
            }
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(QwenBackendRun::Wgpu(build_wgpu_run::<
                    burn::backend::Wgpu,
                    _,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                    |_| {},
                )?))
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Err(unavailable_backend_error("wgpu"))
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(QwenBackendRun::Cuda(build_default_run::<
                    burn::backend::Cuda,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                )?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(unavailable_backend_error("cuda"))
            }
        }
        BackendKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                Ok(QwenBackendRun::Rocm(build_default_run::<
                    burn::backend::Rocm,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                )?))
            }
            #[cfg(not(feature = "rocm"))]
            {
                Err(unavailable_backend_error("rocm"))
            }
        }
        BackendKind::Metal => {
            #[cfg(feature = "metal")]
            {
                Ok(QwenBackendRun::Metal(build_wgpu_run::<
                    burn::backend::Metal,
                    _,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                            device,
                            Default::default(),
                        );
                    },
                )?))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(unavailable_backend_error("metal"))
            }
        }
        BackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Ok(QwenBackendRun::Vulkan(build_wgpu_run::<
                    burn::backend::Vulkan,
                    _,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                            device,
                            Default::default(),
                        );
                    },
                )?))
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(unavailable_backend_error("vulkan"))
            }
        }
        BackendKind::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                Ok(QwenBackendRun::WebGpu(build_wgpu_run::<
                    burn::backend::WebGpu,
                    _,
                >(
                    model_dir,
                    release,
                    request,
                    engine_config,
                    run_config,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::WebGpu>(
                            device,
                            Default::default(),
                        );
                    },
                )?))
            }
            #[cfg(not(feature = "webgpu"))]
            {
                Err(unavailable_backend_error("webgpu"))
            }
        }
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
fn build_default_run<B>(
    model_dir: &Path,
    release: &'static QwenReleaseManifest,
    request: QwenRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<BackendRun<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    build_run_on_device::<B>(
        model_dir,
        release,
        &device,
        request,
        engine_config,
        run_config,
    )
}

#[cfg(any(
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
fn build_wgpu_run<B, F>(
    model_dir: &Path,
    release: &'static QwenReleaseManifest,
    request: QwenRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
    init: F,
) -> Result<BackendRun<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend<Device = burn::backend::wgpu::WgpuDevice>,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = Default::default();
    init(&device);
    build_run_on_device::<B>(
        model_dir,
        release,
        &device,
        request,
        engine_config,
        run_config,
    )
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
fn build_run_on_device<B>(
    model_dir: &Path,
    release: &'static QwenReleaseManifest,
    device: &B::Device,
    request: QwenRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<BackendRun<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    let engine = EngineArtifact::assemble(model_dir, release, device, engine_config)
        .map_err(to_executor_error)?;
    let run = engine
        .start_run(
            request,
            QwenRunConfig {
                max_new_tokens: run_config.max_new_tokens,
                sampling: run_config.sampling,
            },
            device,
        )
        .map_err(to_executor_error)?;
    Ok(BackendRun {
        engine,
        device: device.clone(),
        run,
    })
}

fn unavailable_backend_error(name: &str) -> TtsCoreError {
    to_executor_error(crate::QwenTtsInferenceError::InvalidInput {
        message: format!("backend `{name}` is not compiled in"),
    })
}

fn map_backend(backend: &ComputeBackend) -> BackendKind {
    match backend {
        ComputeBackend::Flex => BackendKind::Flex,
        ComputeBackend::Wgpu => BackendKind::Wgpu,
        ComputeBackend::Cuda => BackendKind::Cuda,
        ComputeBackend::Rocm => BackendKind::Rocm,
        ComputeBackend::Metal => BackendKind::Metal,
        ComputeBackend::Vulkan => BackendKind::Vulkan,
        ComputeBackend::WebGpu => BackendKind::WebGpu,
    }
}

fn map_step(step: QwenRunStep) -> ModelStep {
    let _runtime_step = RunStep {
        generated_steps: step.generated_steps,
        finished: step.finished,
    };
    ModelStep {
        generated_steps: step.generated_steps,
        finished: step.finished,
    }
}

fn map_result(result: FinishedInference) -> SynthesisResult {
    SynthesisResult {
        waveform_pcm: result.waveform_pcm,
        sample_rate: result.sample_rate,
    }
}

fn to_executor_error(error: impl ToString) -> TtsCoreError {
    TtsCoreError::Executor {
        family: "qwen",
        message: error.to_string(),
    }
}
