#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTtsVariant {
    Qwen3Tts12Hz06BCustomVoice,
}

impl QwenTtsVariant {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "qwen3-tts-12hz-0.6b-customvoice" => Some(Self::Qwen3Tts12Hz06BCustomVoice),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BCustomVoice => "qwen3-tts-12hz-0.6b-customvoice",
        }
    }
}
