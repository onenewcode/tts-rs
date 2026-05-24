mod config;
mod init;
mod load;
mod model;
mod remap;
mod verify;

#[cfg(test)]
mod tests;

pub use config::{Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};
pub use load::{LoadedQwen3TtsTalker, load_qwen3_tts_talker};
pub use model::{
    Qwen3TtsAttention, Qwen3TtsCheckpoint, Qwen3TtsDecoderLayer,
    Qwen3TtsTalkerCodePredictorForConditionalGeneration, Qwen3TtsTalkerCodePredictorModel,
    Qwen3TtsTalkerForConditionalGeneration, Qwen3TtsTalkerModel, Qwen3TtsTalkerResizeMlp,
    Qwen3TtsTextMlp,
};
pub use verify::verify_qwen3_tts_talker_weights;
