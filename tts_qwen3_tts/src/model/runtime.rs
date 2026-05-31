use std::sync::Arc;

use burn::tensor::backend::Backend;

use super::{BackendRuntime, LoadedModelOps, Qwen3TtsModelInner};
use crate::{
    Qwen3TtsBackend, Qwen3TtsLoadError, Qwen3TtsPackage, Qwen3TtsProfilingConfig,
    execution::compiler::Qwen3TtsRequestCompiler,
};

pub(super) fn load_backend_runtime(
    package: Qwen3TtsPackage,
    backend: Qwen3TtsBackend,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
) -> Result<Arc<dyn LoadedModelOps>, Qwen3TtsLoadError> {
    match backend {
        Qwen3TtsBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                load_default_backend::<burn::backend::Flex>(package, profiling, compiler)
            }
            #[cfg(not(feature = "flex"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Flex,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                load_wgpu_backend::<burn::backend::Wgpu, _>(package, profiling, compiler, |_| {})
            }
            #[cfg(not(feature = "wgpu"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Wgpu,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                load_default_backend::<burn::backend::Cuda>(package, profiling, compiler)
            }
            #[cfg(not(feature = "cuda"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Cuda,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::Rocm => {
            #[cfg(feature = "rocm")]
            {
                load_default_backend::<burn::backend::Rocm>(package, profiling, compiler)
            }
            #[cfg(not(feature = "rocm"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Rocm,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                load_wgpu_backend::<burn::backend::Metal, _>(package, profiling, compiler, |device| {
                    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                        device,
                        Default::default(),
                    );
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Metal,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                load_wgpu_backend::<burn::backend::Vulkan, _>(
                    package,
                    profiling,
                    compiler,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                            device,
                            Default::default(),
                        );
                    },
                )
            }
            #[cfg(not(feature = "vulkan"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::Vulkan,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
        Qwen3TtsBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                load_wgpu_backend::<burn::backend::WebGpu, _>(
                    package,
                    profiling,
                    compiler,
                    |device| {
                        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::WebGpu>(
                            device,
                            Default::default(),
                        );
                    },
                )
            }
            #[cfg(not(feature = "webgpu"))]
            {
                unavailable_backend_result(
                    Qwen3TtsBackend::WebGpu,
                    package,
                    profiling,
                    compiler,
                )
            }
        }
    }
}

fn load_runtime<B>(
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
    device: &B::Device,
) -> Result<Arc<dyn LoadedModelOps>, Qwen3TtsLoadError>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Clone + Send + Sync + 'static,
{
    Ok(Arc::new(BackendRuntime::new(Qwen3TtsModelInner::<B>::load(
        package, profiling, compiler, device,
    )?)) as Arc<dyn LoadedModelOps>)
}

fn load_default_backend<B>(
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
) -> Result<Arc<dyn LoadedModelOps>, Qwen3TtsLoadError>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Clone + Default + Send + Sync + 'static,
{
    let device = Default::default();
    load_runtime::<B>(package, profiling, compiler, &device)
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
) -> Result<Arc<dyn LoadedModelOps>, Qwen3TtsLoadError>
where
    B: Backend<Device = burn::backend::wgpu::WgpuDevice> + Send + Sync + 'static,
    B::Device: Clone + Send + Sync + 'static,
    F: FnOnce(&burn::backend::wgpu::WgpuDevice),
{
    let device = burn::backend::wgpu::WgpuDevice::default();
    init(&device);
    load_runtime::<B>(package, profiling, compiler, &device)
}

fn unavailable_backend_error(backend: Qwen3TtsBackend) -> Qwen3TtsLoadError {
    Qwen3TtsLoadError::UnavailableBackend {
        backend: backend.label().to_string(),
    }
}

fn unavailable_backend_result<T>(
    backend: Qwen3TtsBackend,
    package: Qwen3TtsPackage,
    profiling: &Qwen3TtsProfilingConfig,
    compiler: Qwen3TtsRequestCompiler,
) -> Result<T, Qwen3TtsLoadError> {
    let _ = (package, profiling, compiler);
    Err(unavailable_backend_error(backend))
}
