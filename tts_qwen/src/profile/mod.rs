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

impl From<&tts_core::SynthesisRequest> for QwenRequest {
    fn from(request: &tts_core::SynthesisRequest) -> Self {
        Self {
            text: request.text.clone(),
            language: request.language.clone(),
            speaker: request.speaker.clone(),
        }
    }
}
