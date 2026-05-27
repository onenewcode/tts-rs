use tts_qwen::{available_backends, resolve_backend};

#[cfg(any(
    feature = "flex",
    feature = "wgpu",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
use std::str::FromStr;
#[cfg(any(
    feature = "flex",
    feature = "wgpu",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
use tts_qwen::BackendKind;

#[test]
fn backend_kind_parses_supported_labels() {
    #[cfg(feature = "flex")]
    assert_eq!(BackendKind::from_str("flex").unwrap(), BackendKind::Flex);

    #[cfg(feature = "wgpu")]
    assert_eq!(BackendKind::from_str("wgpu").unwrap(), BackendKind::Wgpu);

    #[cfg(feature = "cuda")]
    assert_eq!(BackendKind::from_str("cuda").unwrap(), BackendKind::Cuda);

    #[cfg(feature = "rocm")]
    assert_eq!(BackendKind::from_str("rocm").unwrap(), BackendKind::Rocm);

    #[cfg(feature = "metal")]
    assert_eq!(BackendKind::from_str("metal").unwrap(), BackendKind::Metal);

    #[cfg(feature = "vulkan")]
    assert_eq!(
        BackendKind::from_str("vulkan").unwrap(),
        BackendKind::Vulkan
    );

    #[cfg(feature = "webgpu")]
    assert_eq!(
        BackendKind::from_str("webgpu").unwrap(),
        BackendKind::WebGpu
    );
}

#[test]
fn compiled_backends_only_include_supported_runtime_targets() {
    for backend in available_backends() {
        let label = backend.label();
        assert!(matches!(
            label,
            "flex" | "wgpu" | "cuda" | "rocm" | "metal" | "vulkan" | "webgpu"
        ));
    }
}

#[cfg(not(any(
    feature = "flex",
    feature = "wgpu",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
)))]
#[test]
fn resolve_backend_errors_when_no_runtime_backend_is_compiled() {
    let error = resolve_backend(None).unwrap_err().to_string();
    assert!(error.contains("no runtime backend is compiled in"));
}
