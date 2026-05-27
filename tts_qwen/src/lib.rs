//! Local-first Rust inference APIs for on-device speech models.
#![allow(
    clippy::large_enum_variant,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
//!
//! The public surface centers on a model-agnostic local inference core plus
//! model adapters:
//!
//! ```text
//! cli -> local inference core -> model adapter -> waveform -> wav file
//! ```
//!
//! ```no_run
//! use burn::backend::Flex;
//! use tts_qwen::{
//!     CustomVoiceRequest, LocalInferenceCore, LocalInferenceOptions, QwenTtsAdapter,
//! };
//!
//! type Backend = Flex;
//! let device = Default::default();
//! let core = LocalInferenceCore::<Backend, QwenTtsAdapter<Backend>>::load(
//!     "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
//!     &device,
//! )?;
//!
//! let request = CustomVoiceRequest::new("你好，欢迎使用语音合成。");
//! let run = core.infer(&request, &LocalInferenceOptions::default())?;
//! assert!(run.output.sample_rate > 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod adapter;
mod audio_codec;
mod core;
mod frontend;
mod pipeline;
mod shared;
mod talker;

pub use adapter::{
    Qwen3TtsCodecGenerationOutput, Qwen3TtsInferOutput, Qwen3TtsPipelineError,
    Qwen3TtsPipelineLoadReport, QwenTtsAdapter,
};
pub use audio_codec::Qwen3TtsAudioCodecConfig;
pub use core::{
    LocalInferenceCore, LocalInferenceOptions, LocalInferenceProfile, LocalInferenceRun,
    LocalInferenceStageProfile, LocalModelAdapter,
};
pub use frontend::{
    CustomVoiceBatch, CustomVoiceGenerationConfig, CustomVoiceRequest, FrontendOutput,
    Qwen3TtsTextTokenizer, build_custom_voice_prompt, load_custom_voice_generation_config,
};
pub use pipeline::{Qwen3TtsInferOptions, Qwen3TtsPipeline};
pub use shared::error::{Qwen3TtsInferenceError, Qwen3TtsLoadError};
pub use shared::io::{LoadReport, save_wav, write_wav};
pub use shared::paths::{default_workspace_root, find_local_qwen_tts_model_dir};
pub use shared::runtime::sampling::SamplingConfig;
pub use talker::Qwen3TtsTalkerConfig;
