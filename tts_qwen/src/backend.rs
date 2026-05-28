use std::fmt::{self, Display};
use std::str::FromStr;

use crate::QwenTtsInferenceError;

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
