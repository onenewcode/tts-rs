//! On-device Qwen TTS family runtime and model implementations.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

mod arch;
mod backend;
pub mod error;
pub mod io;
mod profile;
pub mod profiling;
mod registry;
mod releases;
mod runtime;

pub use backend::{BackendKind, available_backends, resolve_backend};
pub use error::{
    Qwen3TtsInferenceError, Qwen3TtsLoadError, QwenTtsError, QwenTtsInferenceError,
    QwenTtsLoadError,
};
pub use io::tokenizer::load_qwen3_tts_tokenizer;
pub use io::wav::{save_pcm_wav, save_wav, write_pcm_wav, write_wav};
pub use profile::custom_voice::{
    CustomVoiceGenerationConfig, CustomVoiceRequest, build_custom_voice_prompt,
    load_custom_voice_generation_config,
};
pub use profiling::ProfilingConfig;
pub use registry::register_qwen_family_model;
pub use tts_core::runtime::sampling::SamplingConfig;

pub(crate) use arch::kernels;
pub(crate) use profile::compile as frontend;
