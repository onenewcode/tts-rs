use std::path::Path;

use burn::tensor::backend::Backend;

use crate::Qwen3TtsVerifyError;
use crate::shared::manifest::{
    VerificationArtifacts, WeightVerificationReport, verify_module_weights,
};

use crate::audio_codec::Qwen3TtsAudioCodecCheckpoint;
use crate::shared::io::audio_codec_remap::audio_codec_export_key_remapper;

pub fn verify_qwen3_tts_audio_codec_weights<B: Backend>(
    model: &Qwen3TtsAudioCodecCheckpoint<B>,
    weights_path: impl AsRef<Path>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    verify_module_weights(
        model,
        weights_path,
        Some(audio_codec_export_key_remapper()),
        artifacts,
    )
}
