use std::time::Instant;

mod audio_finalize;
mod backend_runtime;
pub(crate) mod compiler;
pub(crate) mod conditioning;
pub(crate) mod error;
mod loaded_model;
pub(crate) mod profiling;
pub(crate) mod reference_audio;
pub(crate) mod run;
pub(crate) mod session;

use tts_core::driver::ErasedLoadedModel;
use tts_core::{LoadedModelHandle, ModelCapabilities, SynthesisResult};

use crate::{
    BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, Qwen3TtsBackend,
    Qwen3TtsEngineConfig, Qwen3TtsError, Qwen3TtsPackage, Qwen3TtsProfilingConfig,
    Qwen3TtsRunOptions, Qwen3TtsVoiceClonePrompt, QwenRequest,
};

pub(crate) use self::loaded_model::Qwen3TtsLoadedModel;
use self::run::Engine;
// TODO 有很多单层无意义的封装
#[derive(Debug, Clone)]
pub(crate) struct Qwen3LoadedModelInstance {
    model: Qwen3TtsLoadedModel,
    package: Qwen3TtsPackage,
    backend: Qwen3TtsBackend,
    profiling: Qwen3TtsProfilingConfig,
    capabilities: ModelCapabilities,
}

impl Qwen3LoadedModelInstance {
    pub(crate) fn new(
        model: Qwen3TtsLoadedModel,
        package: Qwen3TtsPackage,
        backend: Qwen3TtsBackend,
        profiling: Qwen3TtsProfilingConfig,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            model,
            package,
            backend,
            profiling,
            capabilities,
        }
    }

    pub(crate) fn package(&self) -> &Qwen3TtsPackage {
        &self.package
    }

    pub(crate) fn backend(&self) -> Qwen3TtsBackend {
        self.backend
    }

    pub(crate) fn profiling(&self) -> &Qwen3TtsProfilingConfig {
        &self.profiling
    }

    pub(crate) fn synthesize_audio(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<tts_core::PcmAudio, Qwen3TtsError> {
        let request = self.materialize_request(request)?;
        Engine::new(self.model.clone())
            .synthesize(request, options)
            .map_err(Qwen3TtsError::from)
    }

    pub(crate) fn synthesize_result(
        &self,
        instance_id: u64,
        driver_id: String,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<SynthesisResult, Qwen3TtsError> {
        let started = Instant::now();
        let audio = self.synthesize_audio(request, options)?;
        Ok(SynthesisResult {
            audio,
            instance_id,
            driver_id,
            elapsed: started.elapsed(),
        })
    }

    pub(crate) fn create_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError> {
        self.model
            .create_voice_clone_prompt(&reference)
            .map_err(Qwen3TtsError::from)
    }

    fn materialize_request(&self, request: QwenRequest) -> Result<QwenRequest, Qwen3TtsError> {
        match request {
            QwenRequest::Base(mut request) => {
                if let Some(BaseVoiceCloneConditioning::ReferenceAudio(reference)) =
                    request.voice_clone.take()
                {
                    let prompt = self.create_voice_clone_prompt(reference)?;
                    request.voice_clone = Some(BaseVoiceCloneConditioning::Prompt(prompt));
                }
                Ok(QwenRequest::Base(request))
            }
            QwenRequest::CustomVoice(request) => Ok(QwenRequest::CustomVoice(request)),
        }
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
            model.synthesize_result(instance_id, driver_id, request, options)
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
