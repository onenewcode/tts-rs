use std::fmt::{self, Display};
use std::str::FromStr;

use crate::Qwen3TtsInferenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen3TtsBackend {
    Flex,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    WebGpu,
}

impl Qwen3TtsBackend {
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
}

impl Display for Qwen3TtsBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for Qwen3TtsBackend {
    type Err = Qwen3TtsInferenceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "flex" => Ok(Self::Flex),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            "rocm" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "webgpu" => Ok(Self::WebGpu),
            other => Err(Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "unsupported backend `{other}`; expected one of: {}",
                    Self::ALL_LABELS.join(", ")
                ),
            }),
        }
    }
}
