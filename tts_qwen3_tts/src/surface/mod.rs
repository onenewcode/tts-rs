mod request;

use std::any::type_name;

use tts_core::DriverDescriptor;
use tts_core::DriverRegistry;
use tts_core::PcmAudio;
use tts_core::driver::DriverFactory;
use tts_error::DiagnosticError;

pub use crate::loading::package::{
    Qwen3TtsArtifactsManifest, Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource,
    Qwen3TtsPackage, Qwen3TtsPackageManifest, Qwen3TtsPackageSource,
};
pub use request::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    LanguageSelection, Qwen3TtsVoiceClonePrompt, Qwen3TtsVoiceClonePromptMode, QwenRequest,
};

use crate::execution::Qwen3LoadedModelInstance;

pub const DRIVER_ID: &str = "qwen3_tts";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsEngineConfig {
    pub package: Qwen3TtsPackageSource,
    pub backend: crate::Qwen3TtsBackend,
    pub profiling: crate::Qwen3TtsProfilingConfig,
}

pub type Qwen3TtsLoadOptions = Qwen3TtsEngineConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsRunOptions {
    pub max_new_tokens: Option<usize>,
    pub sampling: crate::SamplingConfig,
}

impl Default for Qwen3TtsRunOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            sampling: crate::SamplingConfig::greedy(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsEngine {
    instance: Qwen3LoadedModelInstance,
}

impl Qwen3TtsEngine {
    pub fn load(config: Qwen3TtsEngineConfig) -> Result<Self, crate::Qwen3TtsLoadError> {
        Ok(Self {
            instance: crate::execution::load_for_engine(&config)?,
        })
    }

    pub fn synthesize(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<PcmAudio, crate::Qwen3TtsError> {
        self.instance.synthesize_audio(request, options)
    }

    pub fn create_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, crate::Qwen3TtsError> {
        self.instance.create_voice_clone_prompt(reference)
    }

    pub fn synthesize_batch<I>(
        &self,
        requests: I,
        options: Qwen3TtsRunOptions,
    ) -> Result<Vec<PcmAudio>, crate::Qwen3TtsError>
    where
        I: IntoIterator<Item = QwenRequest>,
    {
        synthesize_batch_with(requests, |request| {
            self.synthesize(request, options.clone())
        })
    }

    pub fn package(&self) -> &Qwen3TtsPackage {
        self.instance.package()
    }

    pub fn backend(&self) -> crate::Qwen3TtsBackend {
        self.instance.backend()
    }

    pub fn profiling(&self) -> &crate::Qwen3TtsProfilingConfig {
        self.instance.profiling()
    }
}

fn synthesize_batch_with<I, F, T, E>(requests: I, mut synthesize: F) -> Result<Vec<T>, E>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
{
    let mut outputs = Vec::new();
    for request in requests {
        outputs.push(synthesize(request)?);
    }
    Ok(outputs)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3TtsDriver;

impl DriverFactory for Qwen3TtsDriver {
    type Config = Qwen3TtsEngineConfig;

    fn descriptor(&self) -> DriverDescriptor {
        DriverDescriptor::new(
            DRIVER_ID,
            "Qwen3-TTS",
            "Qwen3-TTS local driver",
            type_name::<Qwen3TtsEngineConfig>(),
        )
    }

    fn load(
        &self,
        config: Self::Config,
    ) -> Result<Box<dyn tts_core::driver::ErasedLoadedModel>, DiagnosticError> {
        crate::loading::load_instance(&config)
            .map(|instance| Box::new(instance) as Box<dyn tts_core::driver::ErasedLoadedModel>)
            .map_err(DiagnosticError::from)
    }
}

pub fn register_driver(registry: &mut DriverRegistry) -> Result<(), DiagnosticError> {
    registry.register(Qwen3TtsDriver)
}

#[cfg(test)]
mod tests {
    use super::synthesize_batch_with;
    use tts_core::PcmAudio;

    #[test]
    fn batch_wrapper_preserves_order() {
        let outputs = synthesize_batch_with([1, 2, 3], |value| {
            Ok::<_, &'static str>(PcmAudio {
                pcm_i16: vec![value],
                sample_rate: 24_000,
                channels: 1,
            })
        })
        .unwrap();

        assert_eq!(outputs[0].pcm_i16, vec![1]);
        assert_eq!(outputs[1].pcm_i16, vec![2]);
        assert_eq!(outputs[2].pcm_i16, vec![3]);
    }

    #[test]
    fn batch_wrapper_stops_on_first_error() {
        let mut seen = Vec::new();
        let error = synthesize_batch_with([1, 2, 3], |value| {
            seen.push(value);
            if value == 2 { Err("boom") } else { Ok(value) }
        })
        .unwrap_err();

        assert_eq!(error, "boom");
        assert_eq!(seen, vec![1, 2]);
    }
}
