use std::path::{Path, PathBuf};
use std::sync::Arc;

use tts_core::{
    AudioChunk as CoreAudioChunk, ComputeBackend, ModelCapabilities, ModelRegistry, SessionStep,
    SynthesisEvent, SynthesisOptions, SynthesisRequest, SynthesisResult, TtsCoreError,
    TtsModelAdapter, TtsModelSession,
};

use crate::pipeline::{SessionConfig, SessionHandle, StreamingMode};
use crate::{
    BackendKind, CustomVoiceRequest, EngineConfig, ProfilingConfig, QwenTtsEngine, StepOutcome,
    StreamEvent, resolve_backend,
};

pub struct QwenFamilyAdapter {
    model_dir: PathBuf,
    variant: String,
}

pub fn register_qwen_family_model(
    registry: &mut ModelRegistry,
    model_id: impl Into<String>,
    model_dir: impl AsRef<Path>,
    variant: impl Into<String>,
) -> bool {
    registry
        .register(
            model_id.into(),
            Arc::new(QwenFamilyAdapter::new(model_dir, variant)),
        )
        .is_none()
}

impl QwenFamilyAdapter {
    pub fn new(model_dir: impl AsRef<Path>, variant: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            variant: variant.into(),
        }
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }

    fn resolve_backend(
        &self,
        backend: Option<&ComputeBackend>,
    ) -> Result<BackendKind, TtsCoreError> {
        resolve_backend(backend.map(map_backend)).map_err(to_adapter_error)
    }

    fn build_engine_config(options: &SynthesisOptions) -> EngineConfig {
        EngineConfig {
            codec_chunk_steps: options.chunk_steps,
            profiling: ProfilingConfig {
                enabled: options.profiling,
                per_step: options.profiling,
                stage_summary: true,
                log_topk: 8,
            },
            ..EngineConfig::default()
        }
    }

    fn build_session_config(options: &SynthesisOptions) -> SessionConfig {
        SessionConfig {
            max_new_tokens: options.max_new_tokens,
            sampling: options.sampling.clone(),
            streaming: if options.stream {
                StreamingMode::AudioChunks
            } else {
                StreamingMode::Full
            },
        }
    }
}

impl TtsModelAdapter for QwenFamilyAdapter {
    fn family(&self) -> &'static str {
        "qwen"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_streaming: true,
            supports_custom_voice: true,
        }
    }

    fn start_session(
        &self,
        request: &SynthesisRequest,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsModelSession>, TtsCoreError> {
        let backend = self.resolve_backend(options.backend.as_ref())?;
        let request = CustomVoiceRequest {
            text: request.text.clone(),
            language: request.language.clone(),
            speaker: request.speaker.clone(),
        };
        let engine_config = Self::build_engine_config(options);
        let session_config = Self::build_session_config(options);

        let session =
            match backend {
                BackendKind::Flex => {
                    #[cfg(feature = "flex")]
                    {
                        QwenSession::Flex(build_default_session::<burn::backend::Flex>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                        )?)
                    }
                    #[cfg(not(feature = "flex"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `flex` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Wgpu => {
                    #[cfg(feature = "wgpu")]
                    {
                        QwenSession::Wgpu(build_wgpu_session::<burn::backend::Wgpu, _>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                            |_| {},
                        )?)
                    }
                    #[cfg(not(feature = "wgpu"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `wgpu` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        QwenSession::Cuda(build_default_session::<burn::backend::Cuda>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                        )?)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `cuda` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Rocm => {
                    #[cfg(feature = "rocm")]
                    {
                        QwenSession::Rocm(build_default_session::<burn::backend::Rocm>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                        )?)
                    }
                    #[cfg(not(feature = "rocm"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `rocm` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Metal => {
                    #[cfg(feature = "metal")]
                    {
                        QwenSession::Metal(build_wgpu_session::<burn::backend::Metal, _>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::Metal,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `metal` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::Vulkan => {
                    #[cfg(feature = "vulkan")]
                    {
                        QwenSession::Vulkan(build_wgpu_session::<burn::backend::Vulkan, _>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::Vulkan,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `vulkan` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
                BackendKind::WebGpu => {
                    #[cfg(feature = "webgpu")]
                    {
                        QwenSession::WebGpu(build_wgpu_session::<burn::backend::WebGpu, _>(
                            &self.model_dir,
                            request,
                            engine_config,
                            session_config,
                            |device| {
                                burn::backend::wgpu::init_setup::<
                                    burn::backend::wgpu::graphics::WebGpu,
                                >(device, Default::default());
                            },
                        )?)
                    }
                    #[cfg(not(feature = "webgpu"))]
                    {
                        return Err(to_adapter_error(
                            crate::QwenTtsInferenceError::InvalidInput {
                                message: "backend `webgpu` is not compiled in".to_string(),
                            },
                        ));
                    }
                }
            };

        Ok(Box::new(session))
    }
}

struct EngineSession<B>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    engine: QwenTtsEngine<B>,
    handle: SessionHandle,
}

impl<B> EngineSession<B>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    fn step(&mut self) -> Result<SessionStep, TtsCoreError> {
        match self.engine.step(self.handle).map_err(to_adapter_error)? {
            StepOutcome::Finished => Ok(SessionStep::Finished),
            StepOutcome::MadeProgress | StepOutcome::ProducedEvents => Ok(SessionStep::Running),
        }
    }

    fn drain_events(&mut self) -> Result<Vec<SynthesisEvent>, TtsCoreError> {
        self.engine
            .drain_events(self.handle)
            .map_err(to_adapter_error)
            .map(|events| events.into_iter().map(map_event).collect())
    }

    fn finish(mut self) -> Result<SynthesisResult, TtsCoreError> {
        self.engine
            .finish_session(self.handle)
            .map_err(to_adapter_error)
            .map(|result| SynthesisResult {
                waveform_pcm: result.waveform_pcm,
                sample_rate: result.sample_rate,
            })
    }
}

enum QwenSession {
    #[cfg(feature = "flex")]
    Flex(EngineSession<burn::backend::Flex>),
    #[cfg(feature = "wgpu")]
    Wgpu(EngineSession<burn::backend::Wgpu>),
    #[cfg(feature = "cuda")]
    Cuda(EngineSession<burn::backend::Cuda>),
    #[cfg(feature = "rocm")]
    Rocm(EngineSession<burn::backend::Rocm>),
    #[cfg(feature = "metal")]
    Metal(EngineSession<burn::backend::Metal>),
    #[cfg(feature = "vulkan")]
    Vulkan(EngineSession<burn::backend::Vulkan>),
    #[cfg(feature = "webgpu")]
    WebGpu(EngineSession<burn::backend::WebGpu>),
}

impl TtsModelSession for QwenSession {
    fn step(&mut self) -> Result<SessionStep, TtsCoreError> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(session) => session.step(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(session) => session.step(),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.step(),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session.step(),
            #[cfg(feature = "metal")]
            Self::Metal(session) => session.step(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(session) => session.step(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(session) => session.step(),
        }
    }

    fn drain_events(&mut self) -> Result<Vec<SynthesisEvent>, TtsCoreError> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(session) => session.drain_events(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(session) => session.drain_events(),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.drain_events(),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session.drain_events(),
            #[cfg(feature = "metal")]
            Self::Metal(session) => session.drain_events(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(session) => session.drain_events(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(session) => session.drain_events(),
        }
    }

    fn finish(self: Box<Self>) -> Result<SynthesisResult, TtsCoreError> {
        match *self {
            #[cfg(feature = "flex")]
            Self::Flex(session) => session.finish(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(session) => session.finish(),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.finish(),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session.finish(),
            #[cfg(feature = "metal")]
            Self::Metal(session) => session.finish(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(session) => session.finish(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(session) => session.finish(),
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
fn build_default_session<B>(
    model_dir: &Path,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
) -> Result<EngineSession<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    build_session_on_device::<B>(model_dir, &device, request, engine_config, session_config)
}

#[cfg(any(
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
fn build_wgpu_session<B, F>(
    model_dir: &Path,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
    init: F,
) -> Result<EngineSession<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend<Device = burn::backend::wgpu::WgpuDevice>,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = Default::default();
    init(&device);
    build_session_on_device::<B>(model_dir, &device, request, engine_config, session_config)
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
fn build_session_on_device<B>(
    model_dir: &Path,
    device: &B::Device,
    request: CustomVoiceRequest,
    engine_config: EngineConfig,
    session_config: SessionConfig,
) -> Result<EngineSession<B>, TtsCoreError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    let mut engine =
        QwenTtsEngine::<B>::load(model_dir, device, engine_config).map_err(to_adapter_error)?;
    let handle = engine
        .start_session(request, session_config)
        .map_err(to_adapter_error)?;
    Ok(EngineSession { engine, handle })
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

fn map_event(event: StreamEvent) -> SynthesisEvent {
    match event {
        StreamEvent::CodecChunk { steps } => SynthesisEvent::CodecChunk { steps },
        StreamEvent::AudioChunk(chunk) => SynthesisEvent::AudioChunk(CoreAudioChunk {
            pcm: chunk.pcm,
            sample_rate: chunk.sample_rate,
            is_final: chunk.is_final,
        }),
        StreamEvent::Finished => SynthesisEvent::Finished,
    }
}

fn to_adapter_error(error: impl ToString) -> TtsCoreError {
    TtsCoreError::Adapter {
        model_type: "qwen",
        message: error.to_string(),
    }
}
