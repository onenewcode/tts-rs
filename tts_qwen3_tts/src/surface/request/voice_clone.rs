use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum BaseVoiceCloneConditioning {
    ReferenceAudio(BaseVoiceCloneReferenceAudio),
    Prompt(Qwen3TtsVoiceClonePrompt),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseVoiceCloneReferenceAudio {
    pub path: PathBuf,
    pub transcript: Option<String>,
    pub x_vector_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsVoiceClonePrompt {
    pub speaker_embedding: Vec<f32>,
    pub ref_codec_token_ids: Option<Vec<Vec<i64>>>,
    pub transcript: Option<String>,
    pub mode: Qwen3TtsVoiceClonePromptMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3TtsVoiceClonePromptMode {
    Icl,
    XVectorOnly,
}
