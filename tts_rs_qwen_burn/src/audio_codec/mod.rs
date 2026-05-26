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
//! | `decode_codec_tokens` | Full decoder pipeline: tokens → waveform |
//! | `decode_codec_tokens_single_step` | Convenience for `[batch, num_quantizers]` input |

mod inference;
mod factory {
    pub mod decoder;
    pub mod encoder;
}
mod model {
    pub mod decoder;
    pub mod encoder;
    pub mod wave_decoder;
}
#[cfg(test)]
mod tests;

pub use crate::shared::config::audio_codec::{
    Qwen3TtsAudioCodecConfig, Qwen3TtsAudioCodecDecoderConfig,
    Qwen3TtsAudioCodecEncoderConfig,
};
pub use inference::{decode_codec_tokens, decode_codec_tokens_single_step};
pub use crate::shared::io::audio_codec_load::{
    LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec,
};
pub use crate::shared::nn::activation::{
    Qwen3TtsAudioCodecEmptyModule, AudioCodecLayerScale, AudioCodecSnakeBeta,
};
pub use crate::shared::nn::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d};
pub use model::decoder::{
    Qwen3TtsAudioCodecCheckpoint, Qwen3TtsAudioCodecConvNeXtBlock,
    Qwen3TtsAudioCodecDecoder, Qwen3TtsAudioCodecDecoderAttention,
    Qwen3TtsAudioCodecDecoderCodebook, Qwen3TtsAudioCodecDecoderMlp,
    Qwen3TtsAudioCodecDecoderQuantizer,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantization,
    Qwen3TtsAudioCodecDecoderResidualVectorQuantizer,
    Qwen3TtsAudioCodecDecoderTransformer, Qwen3TtsAudioCodecDecoderTransformerLayer,
    Qwen3TtsAudioCodecDecoderVectorQuantization,
};
pub use model::encoder::{
    Qwen3TtsAudioCodecEncoder, Qwen3TtsAudioCodecEncoderAttention,
    Qwen3TtsAudioCodecEncoderBackbone, Qwen3TtsAudioCodecEncoderBackboneLayer,
    Qwen3TtsAudioCodecEncoderCodebook, Qwen3TtsAudioCodecEncoderConvLayer,
    Qwen3TtsAudioCodecEncoderMlp, Qwen3TtsAudioCodecEncoderQuantizer,
    Qwen3TtsAudioCodecEncoderResidualVectorQuantizer,
    Qwen3TtsAudioCodecEncoderResnetLayer, Qwen3TtsAudioCodecEncoderTransformer,
    Qwen3TtsAudioCodecEncoderTransformerLayer,
    Qwen3TtsAudioCodecEncoderVectorQuantization,
};
pub use model::wave_decoder::{
    Qwen3TtsAudioCodecWaveDecoderConvEntry, Qwen3TtsAudioCodecWaveDecoderEntry,
    Qwen3TtsAudioCodecWaveDecoderResidualUnit,
    Qwen3TtsAudioCodecWaveDecoderUpsampleStage,
};
pub use crate::shared::verify::audio_codec::verify_qwen3_tts_audio_codec_weights;
