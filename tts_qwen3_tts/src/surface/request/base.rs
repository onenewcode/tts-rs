use super::{BaseVoiceCloneConditioning, LanguageSelection};

#[derive(Debug, Clone, PartialEq)]
pub struct BaseRequest {
    pub text: String,
    pub language: LanguageSelection,
    pub voice_clone: Option<BaseVoiceCloneConditioning>,
}

impl BaseRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: LanguageSelection::Auto,
            voice_clone: None,
        }
    }
}
