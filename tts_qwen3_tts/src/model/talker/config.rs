use std::path::Path;

use burn::config::Config;
use serde::{Deserialize, Serialize};

use crate::Qwen3TtsLoadError;

#[derive(Config, Debug)]
pub struct Qwen3TtsTalkerConfig {
    pub code_predictor_config: Qwen3TtsTalkerCodePredictorConfig,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[config(default = "String::from(\"silu\")")]
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[config(default = 32768)]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    #[config(default = "Qwen3TtsTalkerRopeScalingConfig::default()")]
    pub rope_scaling: Qwen3TtsTalkerRopeScalingConfig,
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
    #[config(default = "String::from(\"silu\")")]
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[config(default = 32768)]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[config(default = 10_000.0)]
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub num_code_groups: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsTalkerRopeScalingConfig {
    pub interleaved: bool,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_rope_type", rename = "rope_type")]
    pub rope_type: String,
}

impl Default for Qwen3TtsTalkerRopeScalingConfig {
    fn default() -> Self {
        Self {
            interleaved: true,
            mrope_section: vec![24, 20, 20],
            rope_type: default_rope_type(),
        }
    }
}

fn default_rope_type() -> String {
    "default".to_string()
}

impl Qwen3TtsTalkerConfig {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = path.as_ref().to_path_buf();
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }
}
