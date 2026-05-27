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
pub(crate) mod model;
mod nn;
mod types;

#[cfg(test)]
mod tests;

pub use crate::shared::config::talker::Qwen3TtsTalkerConfig;
pub use crate::shared::io::talker_load::load_qwen3_tts_talker_for_inference;
pub use crate::shared::runtime::cache::KeyValueCache;
pub use crate::shared::runtime::sampling::{SamplingConfig, StoppingRules};
pub use inference::{generate_code_predictor_groups, generate_talker_tokens};
pub use types::{CodePredictorGenerateInput, TalkerGenerateInput};
