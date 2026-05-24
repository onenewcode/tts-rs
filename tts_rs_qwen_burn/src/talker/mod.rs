mod config;
mod cache;
mod init;
mod inference;
mod load;
mod model;
pub mod nn;
mod remap;
mod verify;

#[cfg(test)]
mod tests;

pub use cache::KeyValueCache;
pub use config::{Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};

pub use inference::{
    CodePredictorTeacherForcedInput, CodePredictorTeacherForcedOutput, TalkerForwardInput,
    TalkerForwardOutput, forward_code_predictor_teacher_forced, forward_talker_prefill,
};
pub use load::{LoadedQwen3TtsTalker, load_qwen3_tts_talker, load_qwen3_tts_talker_for_inference};
pub use model::{
    Qwen3TtsCheckpoint, Qwen3TtsTalker, Qwen3TtsTalkerCodePredictor,
    Qwen3TtsTalkerCodePredictorModel, Qwen3TtsTalkerModel,
};
pub use nn::{
    Qwen3RotaryEncoding, Qwen3TtsAttention, Qwen3TtsDecoderLayer, Qwen3TtsTalkerResizeMlp,
    Qwen3TtsTextMlp,
};
pub use verify::verify_qwen3_tts_talker_weights;
