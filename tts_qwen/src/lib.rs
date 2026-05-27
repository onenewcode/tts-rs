//! On-device Qwen TTS inference engine.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

pub mod backend;
pub mod engine;
pub mod error;
pub mod io;
pub mod kernels;
pub mod model;
pub mod profiling;
pub mod runners;
pub mod runtime;
pub mod scheduler;
pub mod session;

pub use backend::{
    BackendKind, available_backends, default_engine_config, default_session_config,
    resolve_backend, run_with_backend,
};
pub use engine::{EngineConfig, FinishedInference, QwenTtsEngine, StepOutcome, StreamEvent};
pub use error::{
    Qwen3TtsInferenceError, Qwen3TtsLoadError, QwenTtsError, QwenTtsInferenceError,
    QwenTtsLoadError,
};
pub use io::paths::{default_workspace_root, find_local_qwen_tts_model_dir};
pub use io::tokenizer::Qwen3TtsTextTokenizer;
pub use io::wav::{save_pcm_wav, save_wav, write_pcm_wav, write_wav};
pub use model::config::audio_codec::Qwen3TtsAudioCodecConfig;
pub use model::config::talker::Qwen3TtsTalkerConfig;
pub use model::load_report::LoadReport;
pub use profiling::ProfilingConfig;
pub use runtime::sampling::SamplingConfig;
pub use session::{
    AudioChunk, CustomVoiceGenerationConfig, CustomVoiceRequest, SessionConfig, SessionHandle,
    StreamingMode, build_custom_voice_prompt, load_custom_voice_generation_config,
};
