use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Qwen3TtsPackageManifest {
    pub format: String,
    pub name: String,
    pub artifacts: Qwen3TtsArtifactsManifest,
    pub generation_config: Qwen3TtsGenerationConfigManifest,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Qwen3TtsArtifactsManifest {
    pub tokenizer: PathBuf,
    pub talker_config: PathBuf,
    pub talker_weights: PathBuf,
    pub codec_config: PathBuf,
    pub codec_weights: PathBuf,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Qwen3TtsGenerationConfigManifest {
    pub do_sample: bool,
    pub repetition_penalty: Option<f32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_new_tokens: usize,
}
