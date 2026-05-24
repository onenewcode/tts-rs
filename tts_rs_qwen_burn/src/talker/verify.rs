use std::path::Path;

use burn::tensor::backend::Backend;

use crate::Qwen3TtsVerifyError;
use crate::manifest::{VerificationArtifacts, WeightVerificationReport, verify_module_weights};

use super::model::Qwen3TtsCheckpoint;
use super::remap::talker_export_key_remapper;

pub fn verify_qwen3_tts_talker_weights<B: Backend>(
    model: &Qwen3TtsCheckpoint<B>,
    weights_path: impl AsRef<Path>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    verify_module_weights(
        model,
        weights_path,
        Some(talker_export_key_remapper()),
        artifacts,
    )
}
