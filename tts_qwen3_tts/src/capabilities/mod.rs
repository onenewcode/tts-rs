use tts_infer::ModelCapabilities;

use crate::loading::ResolvedPackage;

pub(crate) fn project_capabilities(resolved: &ResolvedPackage) -> ModelCapabilities {
    ModelCapabilities::builder()
        .supports_base_synthesis(resolved.compiler.profiles.base.is_some())
        .supports_custom_voice(resolved.compiler.profiles.custom_voice.is_some())
        .supports_voice_clone(
            resolved.compiler.profiles.base.is_some() && resolved.has_speaker_encoder,
        )
        .sample_rate_hz(24_000)
        .channels(1)
        .extension("package_name", resolved.package.name.clone())
        .build()
}
