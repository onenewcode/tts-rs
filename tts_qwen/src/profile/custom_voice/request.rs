use crate::profile::QwenRequest;

#[derive(Debug, Clone)]
pub struct CustomVoiceRequest {
    pub text: String,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

impl CustomVoiceRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            speaker: None,
        }
    }
}

impl From<QwenRequest> for CustomVoiceRequest {
    fn from(request: QwenRequest) -> Self {
        Self {
            text: request.text,
            language: request.language,
            speaker: request.speaker,
        }
    }
}
