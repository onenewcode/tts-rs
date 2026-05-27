//! # Talker — Codec Token Generation
//!
//! This domain handles the autoregressive generation of codec token IDs from text
//! embeddings. It includes:
//!
//! - **TalkerModel**: The main transformer that processes embeddings and produces
//!   codec logits through 28+ decoder layers with M-RoPE attention.
//! - **CodePredictor**: A smaller decoder that expands each generated talker token
//!   into multiple codec groups for the audio codec.
//! - **KV Cache**: Optimized key-value caching for incremental autoregressive decoding.
//! - **Sampling**: Configurable token selection (greedy, temperature, top-k, top-p).
//!
//! ## Key Functions
//!
//! | Function | Purpose |
//! |---|---|
//! | `generate_talker_tokens` | Full autoregressive generation loop |
//! | `generate_code_predictor_groups` | Codec group expansion per time step |
//! | `forward_talker_prefill` | Initial prefill pass (cached) |
//! | `forward_talker_decode_step` | Single-step decode (incremental) |
//! | `sample_token` | Token selection from logits |

mod factory;
mod inference;
mod model;
pub mod nn;
mod types;

#[cfg(test)]
mod tests;

pub use crate::shared::config::talker::{
    Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig,
};
pub use crate::shared::io::talker_load::{
    LoadedQwen3TtsTalker, load_qwen3_tts_talker, load_qwen3_tts_talker_for_inference,
};
pub use crate::shared::runtime::cache::KeyValueCache;

pub use crate::shared::runtime::sampling::{SamplingConfig, StoppingRules};
pub use inference::{
    forward_code_predictor_teacher_forced, forward_talker_decode_step, forward_talker_prefill,
    generate_code_predictor_groups, generate_talker_tokens, sample_token,
};
pub use model::{
    Qwen3TtsCheckpoint, Qwen3TtsTalker, Qwen3TtsTalkerCodePredictor,
    Qwen3TtsTalkerCodePredictorModel, Qwen3TtsTalkerModel,
};
pub use nn::{
    Qwen3RotaryEncoding, Qwen3TtsAttention, Qwen3TtsDecoderLayer, Qwen3TtsTalkerResizeMlp,
    Qwen3TtsTextMlp,
};
pub use types::{
    CodePredictorGenerateInput, CodePredictorGenerateOutput, CodePredictorGenerateStepDiagnostic,
    CodePredictorTeacherForcedInput, CodePredictorTeacherForcedOutput, TalkerDecodeInput,
    TalkerDecodeOutput, TalkerForwardInput, TalkerForwardOutput, TalkerGenerateInput,
    TalkerGenerateOutput, TalkerGenerateStepDiagnostic,
};
