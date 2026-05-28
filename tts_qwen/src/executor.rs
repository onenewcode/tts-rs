use std::path::{Path, PathBuf};
use std::sync::Arc;

use tts_core::{
    ComputeBackend, ModelCapabilities, ModelRegistry, ModelStep, SynthesisOptions,
    SynthesisRequest, SynthesisResult, TtsCoreError, TtsModelExecutor, TtsModelRun,
};

use crate::frontend::CustomVoiceRequest;
use crate::model::variant::QwenTtsVariant;
use crate::{
    BackendKind, EngineConfig, ProfilingConfig, QwenRun, QwenRunConfig, QwenRunStep, QwenTtsEngine,
    resolve_backend,
};

pub(crate) struct QwenFamilyExecutor {
    model_dir: PathBuf,
    variant: QwenTtsVariant,
}

pub fn register_qwen_family_model(
    registry: &mut ModelRegistry,
    model_id: impl Into<String>,
    model_dir: impl AsRef<Path>,
    variant: impl AsRef<str>,
) -> bool {
    let variant = match parse_variant(variant.as_ref()) {
        Ok(variant) => variant,
        Err(_) => return false,
    };
    registry
        .register(
            model_id.into(),
            Arc::new(QwenFamilyExecutor::new(model_dir, variant)),
        )
        .is_none()
}

impl QwenFamilyExecutor {
    fn new(model_dir: impl AsRef<Path>, variant: QwenTtsVariant) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            variant,
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

    fn build_run_config(options: &SynthesisOptions) -> QwenRunConfig {
        QwenRunConfig {
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
        request: &SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsModelRun>, TtsCoreError> {
        let backend = self.resolve_backend(options.backend.as_ref())?;
        let request = CustomVoiceRequest {
            text: request.text.clone(),
            language: request.language.clone(),
            speaker: request.speaker.clone(),
        };
        let engine_config = Self::build_engine_config(options);
        let run_config = Self::build_run_config(options);

        let run =
            match backend {
                BackendKind::Flex => {
                    #[cfg(feature = "flex")]
                    {
                        QwenBackendRun::Flex(build_default_run::<burn::backend::Flex>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                        )?)
                    }
                    #[cfg(not(feature = "flex"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `flex` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Wgpu => {
                    #[cfg(feature = "wgpu")]
                    {
                        QwenBackendRun::Wgpu(build_wgpu_run::<burn::backend::Wgpu, _>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                            |_| {},
                        )?)
                    }
                    #[cfg(not(feature = "wgpu"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `wgpu` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        QwenBackendRun::Cuda(build_default_run::<burn::backend::Cuda>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                        )?)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `cuda` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Rocm => {
                    #[cfg(feature = "rocm")]
                    {
                        QwenBackendRun::Rocm(build_default_run::<burn::backend::Rocm>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                        )?)
                    }
                    #[cfg(not(feature = "rocm"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `rocm` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Metal => {
                    #[cfg(feature = "metal")]
                    {
                        QwenBackendRun::Metal(build_wgpu_run::<burn::backend::Metal, _>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::Metal,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `metal` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Vulkan => {
                    #[cfg(feature = "vulkan")]
                    {
                        QwenBackendRun::Vulkan(build_wgpu_run::<burn::backend::Vulkan, _>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::Vulkan,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `vulkan` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::WebGpu => {
                    #[cfg(feature = "webgpu")]
                    {
                        QwenBackendRun::WebGpu(build_wgpu_run::<burn::backend::WebGpu, _>(
                            &self.model_dir,
                            self.variant,
                            request,
                            engine_config,
                            run_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::WebGpu,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "webgpu"))]
                    {
                        return Err(to_executor_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `webgpu` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
            };

        Ok(Box::new(run))
    }
}

struct BackendRun<B>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    engine: QwenTtsEngine<B>,
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
            .snapshot_audio(&self.run)
            .map_err(to_executor_error)
            .map(|result| SynthesisResult {
                waveform_pcm: result.waveform_pcm,
                sample_rate: result.sample_rate,
            })
    }

    fn finish(self) -> Result<SynthesisResult, TtsCoreError> {
        self.engine
            .finish_run(self.run)
            .map_err(to_executor_error)
            .map(|result| SynthesisResult {
                waveform_pcm: result.waveform_pcm,
                sample_rate: result.sample_rate,
            })
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
    variant: QwenTtsVariant,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    run_config: QwenRunConfig,
) -> Result<BackendRun<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    build_run_on_device::<B>(
        model_dir,
        variant,
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
    variant: QwenTtsVariant,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    run_config: QwenRunConfig,
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
        variant,
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
    variant: QwenTtsVariant,
    device: &B::Device,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    run_config: QwenRunConfig,
) -> Result<BackendRun<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    let engine = QwenTtsEngine::<B>::load(model_dir, device, variant, engine_config)
        .map_err(to_executor_error)?;
    let run = engine
        .start_run(request, run_config)
        .map_err(to_executor_error)?;
    Ok(BackendRun { engine, run })
}

fn parse_variant(value: &str) -> Result<QwenTtsVariant, TtsCoreError> {
    QwenTtsVariant::parse(value).ok_or_else(|| TtsCoreError::Config {
        message: format!(
            "unsupported qwen variant `{value}`; currently supported: {}",
            QwenTtsVariant::Qwen3Tts12Hz06BCustomVoice.label()
        ),
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
    ModelStep {
        generated_steps: step.generated_steps,
        finished: step.finished,
    }
}

fn to_executor_error(error: impl ToString) -> TtsCoreError {
    TtsCoreError::Executor {
        family: "qwen",
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_variant;

    #[test]
    fn rejects_unknown_variant() {
        let err = parse_variant("unknown").expect_err("unknown qwen variant should be rejected");
        assert!(err.to_string().contains("unsupported qwen variant"));
    }
}
