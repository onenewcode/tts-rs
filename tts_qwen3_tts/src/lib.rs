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
pub use request::{BaseRequest, CustomVoiceRequest, LanguageSelection, QwenRequest};
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
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

impl Default for Qwen3TtsRunOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::greedy(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsEngine {
    inner: Engine<Qwen3TtsLoadedModel>,
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
            inner: Engine::new(model),
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
        self.inner
            .synthesize(request, options)
            .map_err(Qwen3TtsError::from)
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
