//! # TTS Inference Engine for Qwen3-TTS
//!
//! A Rust inference pipeline for Qwen3-TTS models built on the [Burn](https://burn.dev)
//! deep learning framework.
//!
//! ## Pipeline
//!
//! ```text
//! config.json → load weights → generate codec tokens → decode waveform → save WAV
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use tts_rs_qwen_burn::*;
//! use burn::backend::Flex;
//!
//! type Backend = Flex;
//! let device = Default::default();
//!
//! // Load models (auto-detects variant from config.json)
//! let talker = load_qwen3_tts_talker_for_inference::<Backend>(
//!     "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/talker", &device)?;
//! let tokenizer = load_qwen3_tts_audio_codec::<Backend>(
//!     "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/audio_codec", &device)?;
//!
//! // Run the pipeline...
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Module Map
//!
//! | Module | Domain | Purpose |
//! |---|---|---|
//! | `talker` | Codec generation | TalkerModel + autoregressive loop + code predictor |
//! | `audio_codec` | Waveform decoding | Decoder + quantizer + upsampling pipeline |
//! | `error` | Shared | Error types |
//! | `manifest` | Shared | Weight manifest and verification |
//! | `paths` | Shared | Model directory discovery |

pub mod shared;
// Backward-compat: old module paths re-export from shared/
pub mod error {
    pub use crate::shared::error::*;
}
pub mod manifest {
    pub use crate::shared::manifest::*;
}
pub mod paths {
    pub use crate::shared::paths::*;
}
pub mod audio_codec;
pub mod frontend;
pub mod talker;

// Backward-compat re-exports from shared/
pub use shared::error::{Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsVerifyError};
pub use shared::manifest::{
    LoadReport, VerificationArtifacts, WeightComparisonReport, WeightManifest, WeightManifestEntry,
    WeightMismatch, WeightVerificationReport,
};
pub use shared::paths::{default_workspace_root, find_local_qwen_tts_model_dir};
pub use audio_codec::{
    LoadedQwen3TtsAudioCodec, Qwen3TtsAudioCodecCheckpoint,
    decode_codec_tokens, decode_codec_tokens_single_step, load_qwen3_tts_audio_codec,
    verify_qwen3_tts_audio_codec_weights,
};
pub use frontend::{
    CustomVoiceBatch, CustomVoiceRequest, FrontendOutput, Qwen3TtsTextTokenizer,
    build_custom_voice_prefill_batch, build_custom_voice_prompt,
    load_custom_voice_generation_config,
};
pub use shared::io::{save_wav, write_wav};
pub use talker::{
    CodePredictorGenerateInput, CodePredictorGenerateOutput,
    CodePredictorGenerateStepDiagnostic, CodePredictorTeacherForcedInput,
    CodePredictorTeacherForcedOutput, KeyValueCache, LoadedQwen3TtsTalker, Qwen3TtsCheckpoint,
    Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig, SamplingConfig,
    StoppingRules, TalkerDecodeInput, TalkerDecodeOutput, TalkerForwardInput, TalkerForwardOutput,
    TalkerGenerateInput, TalkerGenerateOutput, TalkerGenerateStepDiagnostic,
    forward_code_predictor_teacher_forced, forward_talker_decode_step, forward_talker_prefill,
    generate_code_predictor_groups, generate_talker_tokens, load_qwen3_tts_talker,
    load_qwen3_tts_talker_for_inference, sample_token, verify_qwen3_tts_talker_weights,
};
