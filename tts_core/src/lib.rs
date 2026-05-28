mod adapter;
mod error;
mod registry;
pub mod runtime;
pub mod scheduler;
mod service;
mod types;
mod wav;

pub use adapter::{ModelCapabilities, TtsModelAdapter, TtsModelSession};
pub use error::TtsCoreError;
pub use registry::ModelRegistry;
pub use service::TtsService;
pub use types::{
    AudioChunk, ComputeBackend, SessionStep, SynthesisEvent, SynthesisOptions, SynthesisRequest,
    SynthesisResult,
};
pub use wav::save_pcm_wav;
