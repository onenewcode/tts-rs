use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Qwen3TtsPackageManifest {
    pub format: String,
    pub name: String,
    pub artifacts: Qwen3TtsArtifactsManifest,
    pub profiles: Qwen3TtsProfilesManifest,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Qwen3TtsArtifactsManifest {
    pub tokenizer: PathBuf,
    pub talker_config: PathBuf,
    pub talker_weights: PathBuf,
    pub codec_config: PathBuf,
    pub codec_weights: PathBuf,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Qwen3TtsProfilesManifest {
    pub base: Option<Qwen3TtsProfileManifest>,
    pub custom_voice: Option<Qwen3TtsProfileManifest>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Qwen3TtsProfileManifest {
    pub generation_config: PathBuf,
    pub control_config: PathBuf,
}
