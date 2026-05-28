mod error;
mod executor;
mod registry;
pub mod runtime;
pub mod scheduler;
mod service;
mod types;
mod wav;

pub use error::TtsCoreError;
pub use executor::{ModelCapabilities, ModelStep, TtsModelExecutor, TtsModelRun};
pub use registry::ModelRegistry;
pub use service::TtsService;
pub use types::{
    AudioChunk, ComputeBackend, SynthesisEvent, SynthesisOptions, SynthesisRequest, SynthesisResult,
};
pub use wav::save_pcm_wav;
