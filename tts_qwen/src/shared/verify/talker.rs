use std::path::Path;

use burn::tensor::backend::Backend;

use crate::Qwen3TtsVerifyError;
use crate::shared::manifest::{
    VerificationArtifacts, WeightVerificationReport, verify_module_weights,
};

use crate::shared::io::talker_remap::talker_export_key_remapper;
use crate::talker::Qwen3TtsCheckpoint;

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
