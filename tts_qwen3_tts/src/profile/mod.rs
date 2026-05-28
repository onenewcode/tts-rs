pub(crate) mod base;
pub(crate) mod compile;
pub mod custom_voice;
pub(crate) mod model_config;

#[derive(Debug, Clone)]
pub(crate) struct QwenRequest {
    pub text: String,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

impl From<&crate::QwenRequest> for QwenRequest {
    fn from(request: &crate::QwenRequest) -> Self {
        match request {
            crate::QwenRequest::Base(base) => Self {
                text: base.text.clone(),
                language: language_to_option(&base.language),
                speaker: None,
            },
            crate::QwenRequest::CustomVoice(custom_voice) => Self {
                text: custom_voice.text.clone(),
                language: language_to_option(&custom_voice.language),
                speaker: custom_voice.speaker.clone(),
            },
        }
    }
}

fn language_to_option(language: &crate::LanguageSelection) -> Option<String> {
    match language {
        crate::LanguageSelection::Auto => None,
        crate::LanguageSelection::Named(language) => Some(language.clone()),
    }
}
