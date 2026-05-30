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

#[derive(Debug, Deserialize)]
struct ModelConfigWithTalker {
    talker_config: Qwen3TtsTalkerConfig,
}

impl Qwen3TtsTalkerConfig {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = path.as_ref().to_path_buf();
        let raw = std::fs::read_to_string(&path).map_err(|source| {
            Qwen3TtsLoadError::CompilerConfigIo {
                path: path.clone(),
                source,
            }
        })?;

        // Official Qwen3-TTS package configs place talker runtime fields under
        // the top-level `talker_config` key, so we load only that on-disk shape.
        serde_json::from_str::<ModelConfigWithTalker>(&raw)
            .map(|wrapped| wrapped.talker_config)
            .map_err(|source| Qwen3TtsLoadError::CompilerConfigParse { path, source })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::Qwen3TtsTalkerConfig;

    #[test]
    fn load_from_path_reads_nested_talker_config() {
        let path = unique_temp_path("talker-config.json");
        std::fs::write(&path, TALKER_CONFIG_JSON).expect("test config should be writable");

        let config =
            Qwen3TtsTalkerConfig::load_from_path(&path).expect("nested talker config should load");

        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.code_predictor_config.hidden_size, 1024);
        assert_eq!(config.num_code_groups, 16);
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tts-rs-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos()
        ))
    }

    const TALKER_CONFIG_JSON: &str = r#"{
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "talker_config": {
    "attention_bias": false,
    "code_predictor_config": {
      "attention_bias": false,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "intermediate_size": 3072,
      "max_position_embeddings": 65536,
      "num_attention_heads": 16,
      "num_code_groups": 16,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "vocab_size": 2048
    },
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "num_attention_heads": 16,
    "num_code_groups": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "interleaved": true,
      "mrope_section": [24, 20, 20],
      "rope_type": "default"
    },
    "rope_theta": 1000000,
    "text_hidden_size": 2048,
    "text_vocab_size": 151936,
    "vocab_size": 3072
  }
}"#;
}
