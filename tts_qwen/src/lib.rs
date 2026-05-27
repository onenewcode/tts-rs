//! High-level Rust inference APIs for local Qwen3-TTS models.
#![allow(
    clippy::large_enum_variant,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
//!
//! The public surface intentionally centers on the end-to-end pipeline:
//!
//! ```text
//! model dir -> frontend -> talker -> codec tokens -> waveform -> wav file
//! ```
//!
//! ```no_run
//! use burn::backend::Flex;
//! use tts_qwen::{CustomVoiceRequest, Qwen3TtsPipeline, Qwen3TtsSynthesisOptions};
//!
//! type Backend = Flex;
//! let device = Default::default();
//! let pipeline = Qwen3TtsPipeline::<Backend>::load(
//!     "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
//!     &device,
//! )?;
//!
//! let request = CustomVoiceRequest::new("你好，欢迎使用语音合成。");
//! let output = pipeline.synthesize(&request, &Qwen3TtsSynthesisOptions::default())?;
//! assert!(output.sample_rate > 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod audio_codec;
mod frontend;
mod pipeline;
mod shared;
mod talker;

pub use audio_codec::Qwen3TtsAudioCodecConfig;
pub use frontend::{
    CustomVoiceBatch, CustomVoiceGenerationConfig, CustomVoiceRequest, FrontendOutput,
    Qwen3TtsTextTokenizer, build_custom_voice_prompt, load_custom_voice_generation_config,
};
pub use pipeline::{
    Qwen3TtsCodecGenerationOutput, Qwen3TtsPipeline, Qwen3TtsPipelineError,
    Qwen3TtsPipelineLoadReport, Qwen3TtsSynthesisOptions, Qwen3TtsSynthesisOutput,
};
pub use shared::error::{Qwen3TtsInferenceError, Qwen3TtsLoadError};
pub use shared::io::{LoadReport, save_wav, write_wav};
pub use shared::paths::{default_workspace_root, find_local_qwen_tts_model_dir};
pub use shared::runtime::sampling::SamplingConfig;
pub use talker::Qwen3TtsTalkerConfig;
