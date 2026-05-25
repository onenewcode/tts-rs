pub mod error;
pub mod manifest;
pub mod paths;
pub mod speech_tokenizer;
pub mod talker;

pub use error::{Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsVerifyError};
pub use manifest::{
    LoadReport, VerificationArtifacts, WeightComparisonReport, WeightManifest, WeightManifestEntry,
    WeightMismatch, WeightVerificationReport,
};
pub use paths::{default_workspace_root, find_local_qwen_tts_model_dir};
pub use speech_tokenizer::{
    LoadedQwen3TtsSpeechTokenizer, Qwen3TtsSpeechTokenizerCheckpoint,
    load_qwen3_tts_speech_tokenizer, verify_qwen3_tts_speech_tokenizer_weights,
};
pub use talker::{
    CodePredictorGenerateInput, CodePredictorGenerateOutput,
    CodePredictorGenerateStepDiagnostic, CodePredictorTeacherForcedInput,
    CodePredictorTeacherForcedOutput, KeyValueCache, LoadedQwen3TtsTalker, Qwen3TtsCheckpoint,
    Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig, TalkerDecodeInput,
    TalkerDecodeOutput, TalkerForwardInput, TalkerForwardOutput, TalkerGenerateInput,
    TalkerGenerateOutput, TalkerGenerateStepDiagnostic, forward_code_predictor_teacher_forced,
    forward_talker_decode_step, forward_talker_prefill, generate_code_predictor_groups,
    generate_talker_tokens, load_qwen3_tts_talker, load_qwen3_tts_talker_for_inference,
    verify_qwen3_tts_talker_weights,
};
