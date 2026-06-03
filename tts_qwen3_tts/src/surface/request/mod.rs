mod base;
mod custom_voice;
mod language;
mod voice_clone;

pub use base::BaseRequest;
pub use custom_voice::CustomVoiceRequest;
pub use language::LanguageSelection;
pub use voice_clone::{
    BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, Qwen3TtsVoiceClonePrompt,
    Qwen3TtsVoiceClonePromptMode,
};

#[derive(Debug, Clone, PartialEq)]
pub enum QwenRequest {
    Base(BaseRequest),
    CustomVoice(CustomVoiceRequest),
}
