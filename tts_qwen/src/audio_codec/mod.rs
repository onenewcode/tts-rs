//! # Audio Codec — Waveform Decoding
//!
//! This domain converts quantized codec token IDs to raw audio waveform through
//! a multi-stage upsampling pipeline:
//!
//! - **Quantizer**: Residual Vector Quantization (RVQ) codebook lookup (16 layers)
//! - **Pre-conv + Upsample**: Causal transposed convolutions for time expansion
//! - **Decoder Transformer**: 8-layer bidirectional self-attention with RoPE
//! - **Wave Decoder**: SnakeBeta activations + dilated residual units + output conv
//!
//! ## Key Functions
//!
//! | Function | Purpose |
//! |---|---|
//! | `infer` | Internal codec stage: tokens → waveform |

mod inference;
mod factory {
    pub mod decoder;
    pub mod encoder;
}
pub(crate) mod model {
    pub mod decoder;
    pub mod encoder;
    pub mod wave_decoder;
}
#[cfg(test)]
mod tests;

pub use crate::shared::config::audio_codec::Qwen3TtsAudioCodecConfig;
pub use crate::shared::io::audio_codec_load::{
    LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec,
};
pub(crate) use inference::infer;
