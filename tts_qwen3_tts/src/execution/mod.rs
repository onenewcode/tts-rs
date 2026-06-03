use std::time::Instant;

mod audio_finalize;
pub(crate) mod compiler;
pub(crate) mod conditioning;
pub(crate) mod error;
mod loaded_model;
pub(crate) mod profiling;
pub(crate) mod reference_audio;
pub(crate) mod run;
pub(crate) mod session;

use tts_infer::driver::ErasedLoadedModel;
use tts_infer::{LoadedModelHandle, ModelCapabilities, SynthesisResult};

use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsEngineConfig, Qwen3TtsError, Qwen3TtsPackage,
    Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, Qwen3TtsVoiceClonePrompt, QwenRequest,
};

pub(crate) use self::loaded_model::Qwen3TtsLoadedModel;
use self::run::Engine;
#[derive(Debug, Clone)]
pub(crate) struct Qwen3LoadedModelInstance {
    pub(crate) model: Qwen3TtsLoadedModel,
    pub(crate) package: Qwen3TtsPackage,
    pub(crate) profiling: Qwen3TtsProfilingConfig,
    capabilities: ModelCapabilities,
}

impl Qwen3LoadedModelInstance {
    pub(crate) fn new(
        model: Qwen3TtsLoadedModel,
        package: Qwen3TtsPackage,
        profiling: Qwen3TtsProfilingConfig,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            model,
            package,
            profiling,
            capabilities,
        }
    }

    pub(crate) fn synthesize_audio(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<tts_infer::PcmAudio, Qwen3TtsError> {
        Engine::new(self.model.clone())
            .synthesize(request, options)
            .map_err(Qwen3TtsError::from)
    }

    pub(crate) fn create_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError> {
        self.model
            .create_voice_clone_prompt(&reference)
            .map_err(Qwen3TtsError::from)
    }
}

impl ErasedLoadedModel for Qwen3LoadedModelInstance {
    fn driver_id(&self) -> &'static str {
        crate::surface::DRIVER_ID
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub trait Qwen3TtsHandleExt {
    fn synthesize_qwen(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<SynthesisResult, Qwen3TtsError>;

    fn create_qwen_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError>;
}

impl Qwen3TtsHandleExt for LoadedModelHandle {
    fn synthesize_qwen(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<SynthesisResult, Qwen3TtsError> {
        let instance_id = self.instance_id();
        let driver_id = self.driver_id().to_string();
        self.with_model_as::<Qwen3LoadedModelInstance, _, _>(move |model| {
            let started = Instant::now();
            let audio = model.synthesize_audio(request, options)?;
            Ok(SynthesisResult {
                audio,
                instance_id,
                driver_id,
                elapsed: started.elapsed(),
            })
        })?
    }

    fn create_qwen_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError> {
        self.with_model_as::<Qwen3LoadedModelInstance, _, _>(move |model| {
            model.create_voice_clone_prompt(reference)
        })?
    }
}

pub(crate) fn load_for_engine(
    config: &Qwen3TtsEngineConfig,
) -> Result<Qwen3LoadedModelInstance, crate::Qwen3TtsLoadError> {
    crate::loading::load_instance(config)
}
