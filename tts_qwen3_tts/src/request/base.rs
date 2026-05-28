use super::LanguageSelection;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseRequest {
    pub text: String,
    pub language: LanguageSelection,
}

impl BaseRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: LanguageSelection::Auto,
        }
    }
}
