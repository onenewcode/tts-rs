//! On-device Qwen TTS inference engine.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

mod backend;
mod engine;
pub mod error;
pub mod io;
pub mod kernels;
pub mod model;
mod pipeline;
pub mod profiling;
mod provider;
pub mod runners;

pub use backend::{BackendKind, available_backends, resolve_backend};
pub(crate) use engine::{EngineConfig, QwenTtsEngine, StepOutcome, StreamEvent};
pub use error::{
    Qwen3TtsInferenceError, Qwen3TtsLoadError, QwenTtsError, QwenTtsInferenceError,
    QwenTtsLoadError,
};
pub use io::tokenizer::load_qwen3_tts_tokenizer;
pub use io::wav::{save_pcm_wav, save_wav, write_pcm_wav, write_wav};
pub use model::config::audio_codec::Qwen3TtsAudioCodecConfig;
pub use model::config::talker::Qwen3TtsTalkerConfig;
pub use pipeline::{
    CustomVoiceGenerationConfig, CustomVoiceRequest, build_custom_voice_prompt,
    load_custom_voice_generation_config,
};
pub use profiling::ProfilingConfig;
pub use provider::{QwenFamilyAdapter, register_qwen_family_model};
pub use tts_core::runtime::sampling::SamplingConfig;
