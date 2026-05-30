use tts_qwen3_tts::{Qwen3TtsBackend, Qwen3TtsError, Qwen3TtsInferenceError};

pub fn available_backends() -> Vec<Qwen3TtsBackend> {
    [
        cfg!(feature = "flex").then_some(Qwen3TtsBackend::Flex),
        cfg!(feature = "wgpu").then_some(Qwen3TtsBackend::Wgpu),
        cfg!(feature = "cuda").then_some(Qwen3TtsBackend::Cuda),
        cfg!(feature = "rocm").then_some(Qwen3TtsBackend::Rocm),
        cfg!(feature = "metal").then_some(Qwen3TtsBackend::Metal),
        cfg!(feature = "vulkan").then_some(Qwen3TtsBackend::Vulkan),
        cfg!(feature = "webgpu").then_some(Qwen3TtsBackend::WebGpu),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub fn resolve_backend(
    selected: Option<Qwen3TtsBackend>,
) -> Result<Qwen3TtsBackend, Qwen3TtsError> {
    select_backend(selected, &available_backends())
}

fn select_backend(
    selected: Option<Qwen3TtsBackend>,
    available: &[Qwen3TtsBackend],
) -> Result<Qwen3TtsBackend, Qwen3TtsError> {
    match selected {
        Some(backend) if available.contains(&backend) => Ok(backend),
        Some(backend) => Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "backend `{backend}` is not compiled in; available backends: {}",
                format_available_backends(available)
            ),
        }
        .into()),
        None if available.is_empty() => Err(Qwen3TtsInferenceError::InvalidInput {
            message: "no runtime backend is compiled in; enable one of: flex, wgpu, cuda, rocm, metal, vulkan, webgpu"
                .to_string(),
        }
        .into()),
        None if available.len() == 1 => Ok(available[0]),
        None => Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "multiple backends are compiled in; pass --backend one of: {}",
                format_available_backends(available)
            ),
        }
        .into()),
    }
}

fn format_available_backends(backends: &[Qwen3TtsBackend]) -> String {
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
        let selected = select_backend(None, &[Qwen3TtsBackend::Flex]).unwrap();
        assert_eq!(selected, Qwen3TtsBackend::Flex);
    }

    #[test]
    fn select_backend_requires_explicit_choice_with_multiple_backends() {
        let error = select_backend(None, &[Qwen3TtsBackend::Flex, Qwen3TtsBackend::Cuda])
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple backends are compiled in"));
    }
}
