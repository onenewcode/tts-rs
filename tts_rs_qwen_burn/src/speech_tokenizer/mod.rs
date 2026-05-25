//! # Speech Tokenizer — Waveform Decoding
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
    pub mod common;
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

pub use crate::shared::config::tokenizer::{
    Qwen3TtsSpeechTokenizerConfig, Qwen3TtsSpeechTokenizerDecoderConfig,
    Qwen3TtsSpeechTokenizerEncoderConfig,
};
pub use inference::{decode_codec_tokens, decode_codec_tokens_single_step};
pub use crate::shared::io::tokenizer_load::{
    LoadedQwen3TtsSpeechTokenizer, load_qwen3_tts_speech_tokenizer,
};
pub use crate::shared::nn::activation::{
    Qwen3TtsSpeechTokenizerEmptyModule, TokenizerLayerScale, TokenizerSnakeBeta,
};
pub use crate::shared::nn::conv::{TokenizerCausalConv1d, TokenizerCausalTransConv1d};
pub use model::decoder::{
    Qwen3TtsSpeechTokenizerCheckpoint, Qwen3TtsSpeechTokenizerConvNeXtBlock,
    Qwen3TtsSpeechTokenizerDecoder, Qwen3TtsSpeechTokenizerDecoderAttention,
    Qwen3TtsSpeechTokenizerDecoderCodebook, Qwen3TtsSpeechTokenizerDecoderMlp,
    Qwen3TtsSpeechTokenizerDecoderQuantizer,
    Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization,
    Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer,
    Qwen3TtsSpeechTokenizerDecoderTransformer, Qwen3TtsSpeechTokenizerDecoderTransformerLayer,
    Qwen3TtsSpeechTokenizerDecoderVectorQuantization,
};
pub use model::encoder::{
    Qwen3TtsSpeechTokenizerEncoder, Qwen3TtsSpeechTokenizerEncoderAttention,
    Qwen3TtsSpeechTokenizerEncoderBackbone, Qwen3TtsSpeechTokenizerEncoderBackboneLayer,
    Qwen3TtsSpeechTokenizerEncoderCodebook, Qwen3TtsSpeechTokenizerEncoderConvLayer,
    Qwen3TtsSpeechTokenizerEncoderMlp, Qwen3TtsSpeechTokenizerEncoderQuantizer,
    Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer,
    Qwen3TtsSpeechTokenizerEncoderResnetLayer, Qwen3TtsSpeechTokenizerEncoderTransformer,
    Qwen3TtsSpeechTokenizerEncoderTransformerLayer,
    Qwen3TtsSpeechTokenizerEncoderVectorQuantization,
};
pub use model::wave_decoder::{
    Qwen3TtsSpeechTokenizerWaveDecoderConvEntry, Qwen3TtsSpeechTokenizerWaveDecoderEntry,
    Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit,
    Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage,
};
pub use crate::shared::verify::tokenizer::verify_qwen3_tts_speech_tokenizer_weights;
