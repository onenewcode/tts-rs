use std::sync::Arc;

use burn::tensor::backend::Backend;
use tts_infer::{LoadedModel, ModelSession, SessionStep};

use crate::compiler::session_seed::{SessionSeed, materialize_session_seed};
use crate::model::graph::engine::components::decoder::graph::{decode_waveform, waveform_to_pcm};
use crate::model::graph::engine::components::decoder::lowering::DecoderLowering;
use crate::model::graph::engine::components::decoder::weights::{
    LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec,
};
use crate::model::graph::engine::components::generator::graph::runner::TalkerGenerator;
use crate::model::graph::engine::components::generator::weights::{
    LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference,
};
use crate::profiling::with_session_context;
use crate::runtime::sampling::SamplingConfig as RuntimeSamplingConfig;
use crate::{
    Qwen3TtsBackend, Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsPackage,
    Qwen3TtsProfilingConfig, Qwen3TtsRequestCompiler, Qwen3TtsRunOptions, QwenRequest,
};

pub(crate) mod graph;

#[derive(Debug)]
pub(crate) struct Qwen3TtsModelInner<B: Backend> {
    pub(crate) package: Qwen3TtsPackage,
    pub(crate) device: B::Device,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
    pub(crate) talker: LoadedQwen3TtsTalker<B>,
    pub(crate) decoder: LoadedQwen3TtsAudioCodec<B>,
}

impl<B> Qwen3TtsModelInner<B>
where
    B: Backend,
    B::Device: Clone,
{
    fn compile_session_seed(
        &self,
        request: QwenRequest,
    ) -> Result<SessionSeed<B>, Qwen3TtsInferenceError> {
        let condition = self.compiler.compile_request(&request)?;
        materialize_session_seed(
            &condition,
            &self.talker.config.talker_config,
            &self.talker,
            &self.device,
        )
    }

    fn start_generator(
        &self,
        seed: SessionSeed<B>,
        options: Qwen3TtsRunOptions,
    ) -> Result<TalkerGenerator<B>, Qwen3TtsInferenceError> {
        let codec_eos_token_id = seed.codec_eos_token_id;
        let suppress_token_ids = seed.suppress_token_ids.clone();
        TalkerGenerator::start(
            &self.talker.config.talker_config,
            &self.talker,
            &seed,
            map_sampling(&options.sampling),
            options.max_new_tokens,
            Some(codec_eos_token_id),
            suppress_token_ids,
        )
    }

    fn finalize_audio(
        &self,
        run: &TalkerGenerator<B>,
    ) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        let generated = run.finalize()?;
        let waveform = decode_waveform(&self.decoder, generated.codec_token_ids)?;
        let waveform =
            DecoderLowering::lift_output(self.decoder.config.output_sample_rate as u32, waveform)?;
        let pcm = waveform_to_pcm(&waveform)?;
        Ok(tts_infer::PcmAudio {
            pcm_i16: pcm,
            sample_rate: waveform.sample_rate(),
            channels: 1,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Qwen3TtsLoadedModel {
    #[cfg(feature = "flex")]
    Flex(Arc<Qwen3TtsModelInner<burn::backend::Flex>>),
    #[cfg(feature = "wgpu")]
    Wgpu(Arc<Qwen3TtsModelInner<burn::backend::Wgpu>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<Qwen3TtsModelInner<burn::backend::Cuda>>),
    #[cfg(feature = "rocm")]
    Rocm(Arc<Qwen3TtsModelInner<burn::backend::Rocm>>),
    #[cfg(feature = "metal")]
    Metal(Arc<Qwen3TtsModelInner<burn::backend::Metal>>),
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<Qwen3TtsModelInner<burn::backend::Vulkan>>),
    #[cfg(feature = "webgpu")]
    WebGpu(Arc<Qwen3TtsModelInner<burn::backend::WebGpu>>),
}

impl Qwen3TtsLoadedModel {
    pub(crate) fn load(
        package: Qwen3TtsPackage,
        backend: Qwen3TtsBackend,
        profiling: &Qwen3TtsProfilingConfig,
        compiler: Qwen3TtsRequestCompiler,
    ) -> Result<Self, Qwen3TtsLoadError> {
        match backend {
            Qwen3TtsBackend::Flex => {
                #[cfg(feature = "flex")]
                {
                    Ok(Self::Flex(load_default_backend::<burn::backend::Flex>(
                        package, profiling, compiler,
                    )?))
                }
                #[cfg(not(feature = "flex"))]
                {
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    Ok(Self::Wgpu(load_wgpu_backend::<burn::backend::Wgpu, _>(
                        package,
                        profiling,
                        compiler,
                        |_| {},
                    )?))
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Self::Cuda(load_default_backend::<burn::backend::Cuda>(
                        package, profiling, compiler,
                    )?))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    Ok(Self::Rocm(load_default_backend::<burn::backend::Rocm>(
                        package, profiling, compiler,
                    )?))
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    Ok(Self::Metal(load_wgpu_backend::<burn::backend::Metal, _>(
                        package,
                        profiling,
                        compiler,
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
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    Ok(Self::Vulkan(load_wgpu_backend::<burn::backend::Vulkan, _>(
                        package,
                        profiling,
                        compiler,
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
                    Err(unavailable_backend_error(backend))
                }
            }
            Qwen3TtsBackend::WebGpu => {
                #[cfg(feature = "webgpu")]
                {
                    Ok(Self::WebGpu(load_wgpu_backend::<burn::backend::WebGpu, _>(
                        package,
                        profiling,
                        compiler,
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
                    Err(unavailable_backend_error(backend))
                }
            }
        }
    }
}

impl LoadedModel for Qwen3TtsLoadedModel {
    type Request = QwenRequest;
    type RunOptions = Qwen3TtsRunOptions;
    type Session = Qwen3TtsSession;
    type Error = Qwen3TtsInferenceError;

    fn start_session(
        &self,
        request: Self::Request,
        options: Self::RunOptions,
    ) -> Result<Self::Session, Self::Error> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Flex),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Wgpu),
            #[cfg(feature = "cuda")]
            Self::Cuda(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Cuda),
            #[cfg(feature = "rocm")]
            Self::Rocm(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Rocm),
            #[cfg(feature = "metal")]
            Self::Metal(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Metal),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::Vulkan),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(inner) => start_backend_session(inner, request, options).map(Qwen3TtsSession::WebGpu),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Qwen3TtsSession {
    #[cfg(feature = "flex")]
    Flex(SessionImpl<burn::backend::Flex>),
    #[cfg(feature = "wgpu")]
    Wgpu(SessionImpl<burn::backend::Wgpu>),
    #[cfg(feature = "cuda")]
    Cuda(SessionImpl<burn::backend::Cuda>),
    #[cfg(feature = "rocm")]
    Rocm(SessionImpl<burn::backend::Rocm>),
    #[cfg(feature = "metal")]
    Metal(SessionImpl<burn::backend::Metal>),
    #[cfg(feature = "vulkan")]
    Vulkan(SessionImpl<burn::backend::Vulkan>),
    #[cfg(feature = "webgpu")]
    WebGpu(SessionImpl<burn::backend::WebGpu>),
}

impl ModelSession for Qwen3TtsSession {
    type Error = Qwen3TtsInferenceError;

    fn step(&mut self) -> Result<SessionStep, Self::Error> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(session) => step_impl(session),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(session) => step_impl(session),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => step_impl(session),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => step_impl(session),
            #[cfg(feature = "metal")]
            Self::Metal(session) => step_impl(session),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(session) => step_impl(session),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(session) => step_impl(session),
        }
    }

    fn finish(self) -> Result<tts_infer::PcmAudio, Self::Error> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex(session) => finish_impl(session),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(session) => finish_impl(session),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => finish_impl(session),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => finish_impl(session),
            #[cfg(feature = "metal")]
            Self::Metal(session) => finish_impl(session),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(session) => finish_impl(session),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(session) => finish_impl(session),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SessionImpl<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
    run: TalkerGenerator<B>,
    session_id: usize,
}

fn start_backend_session<B>(
    inner: &Arc<Qwen3TtsModelInner<B>>,
    request: QwenRequest,
    options: Qwen3TtsRunOptions,
) -> Result<SessionImpl<B>, Qwen3TtsInferenceError>
where
    B: Backend,
    B::Device: Clone,
{
    let inner = Arc::clone(inner);
    let seed = inner.compile_session_seed(request)?;
    let run = inner.start_generator(seed, options)?;
    Ok(SessionImpl {
        inner,
        run,
        session_id: 0,
    })
}

fn step_impl<B>(session: &mut SessionImpl<B>) -> Result<SessionStep, Qwen3TtsInferenceError>
where
    B: Backend,
    B::Device: Clone,
{
    let step_idx = session.run.step_idx();
    let step_result = with_session_context(session.session_id, step_idx, || {
        session.run.step(&session.inner.talker)
    })?;
    match step_result {
        Some(step) if step.finished => Ok(SessionStep::Finished),
        Some(_) => Ok(SessionStep::Advanced),
        None => Ok(SessionStep::Finished),
    }
}

fn finish_impl<B>(session: SessionImpl<B>) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError>
where
    B: Backend,
    B::Device: Clone,
{
    session.inner.finalize_audio(&session.run)
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
fn load_default_backend<B>(
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
) -> Result<Arc<Qwen3TtsModelInner<B>>, Qwen3TtsLoadError>
where
    B: Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    load_model_inner::<B>(package, profiling, compiler, &device)
}

#[cfg(any(
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
fn load_wgpu_backend<B, F>(
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
    init: F,
) -> Result<Arc<Qwen3TtsModelInner<B>>, Qwen3TtsLoadError>
where
    B: Backend<Device = burn::backend::wgpu::WgpuDevice>,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = Default::default();
    init(&device);
    load_model_inner::<B>(package, profiling, compiler, &device)
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
fn load_model_inner<B>(
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
    device: &B::Device,
) -> Result<Arc<Qwen3TtsModelInner<B>>, Qwen3TtsLoadError>
where
    B: Backend,
    B::Device: Clone,
{
    crate::profiling::configure(profiling);
    let talker = load_qwen3_tts_talker_for_inference::<B>(
        &package.talker_config_path,
        &package.talker_weights_path,
        device,
    )?;
    let decoder = load_qwen3_tts_audio_codec::<B>(
        &package.codec_config_path,
        &package.codec_weights_path,
        device,
    )?;
    Ok(Arc::new(Qwen3TtsModelInner {
        package,
        device: device.clone(),
        compiler,
        talker,
        decoder,
    }))
}

fn unavailable_backend_error(backend: Qwen3TtsBackend) -> Qwen3TtsLoadError {
    Qwen3TtsLoadError::UnavailableBackend {
        backend: backend.label().to_string(),
    }
}

fn map_sampling(sampling: &crate::SamplingConfig) -> RuntimeSamplingConfig {
    RuntimeSamplingConfig {
        do_sample: sampling.do_sample,
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        seed: sampling.seed,
        repetition_penalty: sampling.repetition_penalty,
    }
}
