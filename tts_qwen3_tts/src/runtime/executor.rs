use std::path::Path;

use crate::arch::engine::assembly::{
    EngineArtifact, FinishedInference, QwenRun, QwenRunConfig, QwenRunStep,
};
use crate::profile::QwenRequest as RuntimeRequest;
use crate::releases::{QwenProfile, QwenReleaseManifest, release_for_profile};
use crate::runtime::types::{EngineConfig, RunConfig, RunStep};
use crate::{
    Qwen3TtsBackend, Qwen3TtsInferenceError, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions,
    SamplingConfig,
};

pub(crate) fn start_model_run(
    model_dir: &Path,
    profile: QwenProfile,
    backend: Qwen3TtsBackend,
    request: &crate::QwenRequest,
    profiling: &Qwen3TtsProfilingConfig,
    options: &Qwen3TtsRunOptions,
) -> Result<QwenModelRun, Qwen3TtsInferenceError> {
    let release = release_for_profile(profile);
    let request = RuntimeRequest::from(request);
    let engine_config = EngineConfig {
        profiling: profiling.clone(),
    };
    let run_config = RunConfig {
        max_new_tokens: options.max_new_tokens,
        sampling: map_sampling(&options.sampling),
    };

    build_release_backend_run(model_dir, release, backend, request, engine_config, run_config)
}

pub(crate) enum QwenModelRun {
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

impl core::fmt::Debug for QwenModelRun {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("QwenModelRun(..)")
    }
}

impl QwenModelRun {
    pub(crate) fn step(&mut self) -> Result<RunStep, Qwen3TtsInferenceError> {
        let step = match self {
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
        }?;
        Ok(map_step(step))
    }

    pub(crate) fn finish(self) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        let result = match self {
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
        }?;
        Ok(map_result(result))
    }
}

pub(crate) struct BackendRun<B>
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
    fn advance(&mut self) -> Result<QwenRunStep, Qwen3TtsInferenceError> {
        self.engine
            .step_run(&mut self.run)
            .map_err(to_inference_error)
    }

    fn finish(self) -> Result<FinishedInference, Qwen3TtsInferenceError> {
        self.engine
            .finish_run(self.run, &self.device)
            .map_err(to_inference_error)
    }
}

fn build_release_backend_run(
    model_dir: &Path,
    release: &'static QwenReleaseManifest,
    backend: Qwen3TtsBackend,
    request: RuntimeRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<QwenModelRun, Qwen3TtsInferenceError> {
    match backend {
        Qwen3TtsBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                Ok(QwenModelRun::Flex(build_default_run::<burn::backend::Flex>(
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
        Qwen3TtsBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(QwenModelRun::Wgpu(build_wgpu_run::<
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
        Qwen3TtsBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(QwenModelRun::Cuda(build_default_run::<burn::backend::Cuda>(
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
        Qwen3TtsBackend::Rocm => {
            #[cfg(feature = "rocm")]
            {
                Ok(QwenModelRun::Rocm(build_default_run::<burn::backend::Rocm>(
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
        Qwen3TtsBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                Ok(QwenModelRun::Metal(build_wgpu_run::<
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
        Qwen3TtsBackend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Ok(QwenModelRun::Vulkan(build_wgpu_run::<
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
        Qwen3TtsBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                Ok(QwenModelRun::WebGpu(build_wgpu_run::<
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
    request: RuntimeRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<BackendRun<B>, Qwen3TtsInferenceError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone + Default,
{
    let device = Default::default();
    build_run_on_device::<B>(model_dir, release, &device, request, engine_config, run_config)
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
    request: RuntimeRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
    init: F,
) -> Result<BackendRun<B>, Qwen3TtsInferenceError>
where
    B: burn::tensor::backend::Backend<Device = burn::backend::wgpu::WgpuDevice>,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = Default::default();
    init(&device);
    build_run_on_device::<B>(model_dir, release, &device, request, engine_config, run_config)
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
    request: RuntimeRequest,
    engine_config: EngineConfig,
    run_config: RunConfig,
) -> Result<BackendRun<B>, Qwen3TtsInferenceError>
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
{
    let engine =
        EngineArtifact::assemble(model_dir, release, device, engine_config).map_err(to_inference_error)?;
    let run = engine
        .start_run(
            request,
            QwenRunConfig {
                max_new_tokens: run_config.max_new_tokens,
                sampling: run_config.sampling,
            },
            device,
        )
        .map_err(to_inference_error)?;
    Ok(BackendRun {
        engine,
        device: device.clone(),
        run,
    })
}

fn unavailable_backend_error(name: &str) -> Qwen3TtsInferenceError {
    Qwen3TtsInferenceError::InvalidInput {
        message: format!("backend `{name}` is not compiled in"),
    }
}

fn map_sampling(sampling: &SamplingConfig) -> crate::runtime::sampling::SamplingConfig {
    crate::runtime::sampling::SamplingConfig {
        do_sample: sampling.do_sample,
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        seed: sampling.seed,
        repetition_penalty: sampling.repetition_penalty,
    }
}

fn map_step(step: QwenRunStep) -> RunStep {
    RunStep {
        generated_steps: step.generated_steps,
        finished: step.finished,
    }
}

fn map_result(result: FinishedInference) -> tts_infer::PcmAudio {
    tts_infer::PcmAudio {
        pcm_i16: result.waveform_pcm,
        sample_rate: result.sample_rate,
        channels: 1,
    }
}

fn to_inference_error(error: impl ToString) -> Qwen3TtsInferenceError {
    Qwen3TtsInferenceError::InvalidInput {
        message: error.to_string(),
    }
}
