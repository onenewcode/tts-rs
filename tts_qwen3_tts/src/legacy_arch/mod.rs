use std::path::Path;

use crate::Qwen3TtsInferenceError;
use crate::profile::model_config::GenerationConfig;
use crate::releases::QwenProfile;

pub(crate) mod engine;
pub mod kernels;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenArchitectureId {
    Qwen3Tts,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenArchitectureDescriptor {
    pub(crate) id: QwenArchitectureId,
    pub(crate) label: &'static str,
    pub(crate) load_generation_config:
        fn(&Path, QwenProfile) -> Result<GenerationConfig, Qwen3TtsInferenceError>,
}

pub(crate) static QWEN3_TTS_ARCH: QwenArchitectureDescriptor = QwenArchitectureDescriptor {
    id: QwenArchitectureId::Qwen3Tts,
    label: "qwen3_tts",
    load_generation_config: load_qwen3_tts_generation_config,
};

fn load_qwen3_tts_generation_config(
    model_dir: &Path,
    profile: QwenProfile,
) -> Result<GenerationConfig, Qwen3TtsInferenceError> {
    match profile {
        QwenProfile::Base => crate::profile::base::config::load_base_generation_config(model_dir),
        QwenProfile::CustomVoice => {
            crate::profile::custom_voice::load_custom_voice_generation_config(model_dir)
        }
    }
}
