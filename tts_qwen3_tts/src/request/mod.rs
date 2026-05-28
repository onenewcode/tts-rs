mod base;
mod custom_voice;
mod language;

pub use base::BaseRequest;
pub use custom_voice::CustomVoiceRequest;
pub use language::LanguageSelection;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenRequest {
    Base(BaseRequest),
    CustomVoice(CustomVoiceRequest),
}
