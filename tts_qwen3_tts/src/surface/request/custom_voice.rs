use super::LanguageSelection;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomVoiceRequest {
    pub text: String,
    pub language: LanguageSelection,
    pub speaker: Option<String>,
    pub instruct: Option<String>,
}

impl CustomVoiceRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: LanguageSelection::Auto,
            speaker: None,
            instruct: None,
        }
    }
}
