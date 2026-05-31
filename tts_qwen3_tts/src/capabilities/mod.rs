use tts_core::ModelCapabilities;

use crate::{
    Qwen3TtsBackend, Qwen3TtsPackage, execution::compiler::Qwen3TtsRequestCompiler,
    model::Qwen3TtsLoadedModel,
};

pub(crate) fn project_capabilities(
    package: &Qwen3TtsPackage,
    backend: Qwen3TtsBackend,
    compiler: &Qwen3TtsRequestCompiler,
    model: &Qwen3TtsLoadedModel,
) -> ModelCapabilities {
    ModelCapabilities::builder()
        .supports_base_synthesis(compiler.profiles.base.is_some())
        .supports_custom_voice(compiler.profiles.custom_voice.is_some())
        .supports_voice_clone(model.supports_voice_clone())
        .sample_rate_hz(24_000)
        .channels(1)
        .extension("package_name", package.name.clone())
        .extension("backend", backend.label())
        .build()
}
