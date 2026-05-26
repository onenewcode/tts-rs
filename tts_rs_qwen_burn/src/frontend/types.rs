use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

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

#[derive(Debug, Clone)]
pub struct CustomVoiceBatch {
    pub requests: Vec<CustomVoiceRequest>,
}

impl CustomVoiceBatch {
    pub fn single(request: CustomVoiceRequest) -> Self {
        Self {
            requests: vec![request],
        }
    }
}

#[derive(Debug)]
pub struct FrontendOutput<B: Backend> {
    pub text_token_ids: Vec<Vec<i64>>,
    pub codec_prefix_ids: Vec<Vec<i64>>,
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
}
