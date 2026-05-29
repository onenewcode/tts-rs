mod backend;
mod compiler;
mod error;
mod io;
mod model;
mod package;
mod profiling;
mod request;
mod runtime;
mod sampling;

use tts_infer::{Engine, PcmAudio};

use compiler::Qwen3TtsRequestCompiler;
use model::Qwen3TtsLoadedModel;

pub use backend::{Qwen3TtsBackend, available_backends, resolve_backend};
pub use error::{Qwen3TtsError, Qwen3TtsInferenceError, Qwen3TtsLoadError};
pub use package::{
    Qwen3TtsArtifactsManifest, Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource,
    Qwen3TtsPackage, Qwen3TtsPackageManifest, Qwen3TtsPackageSource,
};
pub use profiling::Qwen3TtsProfilingConfig;
pub use request::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    LanguageSelection, Qwen3TtsVoiceClonePrompt, Qwen3TtsVoiceClonePromptMode, QwenRequest,
};
pub use sampling::SamplingConfig;

pub(crate) use model::graph::kernels;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsEngineConfig {
    pub package: Qwen3TtsPackageSource,
    pub backend: Qwen3TtsBackend,
    pub profiling: Qwen3TtsProfilingConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsRunOptions {
    pub max_new_tokens: Option<usize>,
    pub sampling: SamplingConfig,
}

impl Default for Qwen3TtsRunOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            sampling: SamplingConfig::greedy(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsEngine {
    inner: Engine<Qwen3TtsLoadedModel>,
    model: Qwen3TtsLoadedModel,
    package: Qwen3TtsPackage,
    backend: Qwen3TtsBackend,
    profiling: Qwen3TtsProfilingConfig,
}

impl Qwen3TtsEngine {
    pub fn load(config: Qwen3TtsEngineConfig) -> Result<Self, Qwen3TtsLoadError> {
        let package = Qwen3TtsPackage::load(&config.package)?;
        let compiler = Qwen3TtsRequestCompiler::load(&package)?;
        let model = Qwen3TtsLoadedModel::load(
            package.clone(),
            config.backend,
            &config.profiling,
            compiler,
        )?;
        Ok(Self {
            inner: Engine::new(model.clone()),
            model,
            package,
            backend: config.backend,
            profiling: config.profiling,
        })
    }

    pub fn synthesize(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<PcmAudio, Qwen3TtsError> {
        let request = self.materialize_request(request)?;
        self.inner
            .synthesize(request, options)
            .map_err(Qwen3TtsError::from)
    }

    pub fn create_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError> {
        self.model
            .create_voice_clone_prompt(&reference)
            .map_err(Qwen3TtsError::from)
    }

    pub fn synthesize_batch<I>(
        &self,
        requests: I,
        options: Qwen3TtsRunOptions,
    ) -> Result<Vec<PcmAudio>, Qwen3TtsError>
    where
        I: IntoIterator<Item = QwenRequest>,
    {
        synthesize_batch_with(requests, |request| {
            self.synthesize(request, options.clone())
        })
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

    pub fn package(&self) -> &Qwen3TtsPackage {
        &self.package
    }

    pub fn backend(&self) -> Qwen3TtsBackend {
        self.backend
    }

    pub fn profiling(&self) -> &Qwen3TtsProfilingConfig {
        &self.profiling
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

#[cfg(test)]
mod tests {
    use super::synthesize_batch_with;
    use tts_infer::PcmAudio;

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
