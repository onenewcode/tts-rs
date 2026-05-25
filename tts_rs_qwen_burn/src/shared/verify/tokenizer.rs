use std::path::Path;

use burn::tensor::backend::Backend;

use crate::Qwen3TtsVerifyError;
use crate::shared::manifest::{VerificationArtifacts, WeightVerificationReport, verify_module_weights};

use crate::speech_tokenizer::Qwen3TtsSpeechTokenizerCheckpoint;
use crate::shared::io::tokenizer_remap::speech_tokenizer_export_key_remapper;

pub fn verify_qwen3_tts_speech_tokenizer_weights<B: Backend>(
    model: &Qwen3TtsSpeechTokenizerCheckpoint<B>,
    weights_path: impl AsRef<Path>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    verify_module_weights(
        model,
        weights_path,
        Some(speech_tokenizer_export_key_remapper()),
        artifacts,
    )
}
