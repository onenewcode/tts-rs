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

#[derive(Debug)]
pub struct CompiledRequest<B: Backend> {
    pub text_token_ids: Vec<i64>,
    pub codec_prefix_ids: Vec<i64>,
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub trailing_text_hidden: Tensor<B, 3>,
    pub tts_pad_embed: Tensor<B, 3>,
}
