use std::path::Path;

use burn::config::Config;

use crate::Qwen3TtsLoadError;

#[derive(Config, Debug)]
pub struct Qwen3TtsConfig {
    pub talker_config: Qwen3TtsTalkerConfig,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsTalkerConfig {
    pub code_predictor_config: Qwen3TtsTalkerCodePredictorConfig,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub num_code_groups: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsTalkerCodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub num_code_groups: usize,
}

impl Qwen3TtsConfig {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = model_dir.as_ref().join("config.json");
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }
}
