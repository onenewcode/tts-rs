mod backend;
mod capabilities;
mod error;
mod execution;
mod io;
mod loading;
mod model;
mod profiling;
mod runtime;
mod sampling;
mod surface;

pub use backend::{Qwen3TtsBackend, available_backends, resolve_backend};
pub use error::{Qwen3TtsError, Qwen3TtsInferenceError, Qwen3TtsLoadError};
pub use execution::Qwen3TtsHandleExt;
pub use profiling::Qwen3TtsProfilingConfig;
pub use sampling::SamplingConfig;
pub use surface::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    DRIVER_ID as QWEN3_TTS_DRIVER_ID, LanguageSelection, Qwen3TtsArtifactsManifest, Qwen3TtsDriver,
    Qwen3TtsEngine, Qwen3TtsEngineConfig, Qwen3TtsGenerationConfigManifest,
    Qwen3TtsGenerationConfigSource, Qwen3TtsLoadOptions, Qwen3TtsPackage, Qwen3TtsPackageManifest,
    Qwen3TtsPackageSource, Qwen3TtsRunOptions, Qwen3TtsVoiceClonePrompt,
    Qwen3TtsVoiceClonePromptMode, QwenRequest, register_driver,
};

pub(crate) use model::graph::kernels;
