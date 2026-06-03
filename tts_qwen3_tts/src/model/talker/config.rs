use std::path::Path;

use burn::config::Config;
use burn::nn::{EmbeddingConfig, LinearConfig, RmsNormConfig};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::Qwen3TtsLoadError;
use crate::model::talker::network::attention::Qwen3TtsAttention;
use crate::model::talker::network::layer::Qwen3TtsDecoderLayer;
use crate::model::talker::network::mlp::{Qwen3TtsTalkerResizeMlp, Qwen3TtsTextMlp};
use crate::model::talker::network::rope::{Qwen3RotaryEncoding, Qwen3StandardRotaryEncoding};
use crate::model::talker::network::{
    Qwen3TtsTalker, Qwen3TtsTalkerCodePredictor, Qwen3TtsTalkerCodePredictorModel,
    Qwen3TtsTalkerModel, Qwen3TtsTalkerModelBundle,
};

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
    pub rope_theta: f32,
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
    pub rope_theta: f32,
    pub attention_bias: bool,
    pub num_code_groups: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsTalkerRopeScalingConfig {
    pub interleaved: bool,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_rope_type")]
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

    pub fn init_model_bundle<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsTalkerModelBundle<B> {
        Qwen3TtsTalkerModelBundle {
            talker: self.init(device),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3TtsTalker<B> {
        Qwen3TtsTalker {
            model: self.init_model(device),
            text_projection: self.init_text_projection(device),
            codec_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
            code_predictor: self.init_code_predictor(device),
            mrope: Qwen3RotaryEncoding::new(
                self.head_dim,
                self.rope_theta,
                self.rope_scaling.mrope_section.clone(),
                device,
            ),
        }
    }

    fn init_model<B: Backend>(&self, device: &B::Device) -> Qwen3TtsTalkerModel<B> {
        Qwen3TtsTalkerModel {
            codec_embedding: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers: (0..self.num_hidden_layers)
                .map(|_| self.init_decoder_layer(device))
                .collect(),
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            text_embedding: EmbeddingConfig::new(self.text_vocab_size, self.text_hidden_size)
                .init(device),
        }
    }

    fn init_decoder_layer<B: Backend>(&self, device: &B::Device) -> Qwen3TtsDecoderLayer<B> {
        Qwen3TtsDecoderLayer {
            self_attn: self.init_attention(device),
            mlp: self.init_mlp(device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }

    fn init_attention<B: Backend>(&self, device: &B::Device) -> Qwen3TtsAttention<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsAttention {
            q_proj: LinearConfig::new(self.hidden_size, q_out)
                .with_bias(self.attention_bias)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, kv_out)
                .with_bias(self.attention_bias)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, kv_out)
                .with_bias(self.attention_bias)
                .init(device),
            o_proj: LinearConfig::new(q_out, self.hidden_size)
                .with_bias(self.attention_bias)
                .init(device),
            q_norm: RmsNormConfig::new(self.head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(self.head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }

    fn init_mlp<B: Backend>(&self, device: &B::Device) -> Qwen3TtsTextMlp<B> {
        Qwen3TtsTextMlp {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    fn init_text_projection<B: Backend>(&self, device: &B::Device) -> Qwen3TtsTalkerResizeMlp<B> {
        Qwen3TtsTalkerResizeMlp {
            linear_fc1: LinearConfig::new(self.text_hidden_size, self.text_hidden_size)
                .with_bias(true)
                .init(device),
            linear_fc2: LinearConfig::new(self.text_hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    fn init_code_predictor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsTalkerCodePredictor<B> {
        let code_predictor = &self.code_predictor_config;
        let num_heads = self.num_code_groups.saturating_sub(1);
        let projection = if code_predictor.hidden_size == self.hidden_size {
            None
        } else {
            Some(
                LinearConfig::new(self.hidden_size, code_predictor.hidden_size)
                    .with_bias(true)
                    .init(device),
            )
        };

        Qwen3TtsTalkerCodePredictor {
            model: code_predictor.init_model(self.hidden_size, device),
            lm_head: (0..num_heads)
                .map(|_| {
                    LinearConfig::new(code_predictor.hidden_size, code_predictor.vocab_size)
                        .with_bias(false)
                        .init(device)
                })
                .collect(),
            small_to_mtp_projection: projection,
            rope: Qwen3StandardRotaryEncoding::new(
                code_predictor.head_dim,
                code_predictor.rope_theta,
                device,
            ),
        }
    }
}

impl Qwen3TtsTalkerCodePredictorConfig {
    fn init_model<B: Backend>(
        &self,
        embedding_dim: usize,
        device: &B::Device,
    ) -> Qwen3TtsTalkerCodePredictorModel<B> {
        let num_heads = self.num_code_groups.saturating_sub(1);

        Qwen3TtsTalkerCodePredictorModel {
            codec_embedding: (0..num_heads)
                .map(|_| EmbeddingConfig::new(self.vocab_size, embedding_dim).init(device))
                .collect(),
            layers: (0..self.num_hidden_layers)
                .map(|_| self.init_decoder_layer(device))
                .collect(),
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }

    fn init_decoder_layer<B: Backend>(&self, device: &B::Device) -> Qwen3TtsDecoderLayer<B> {
        let q_out = self.num_attention_heads * self.head_dim;
        let kv_out = self.num_key_value_heads * self.head_dim;

        Qwen3TtsDecoderLayer {
            self_attn: Qwen3TtsAttention {
                q_proj: LinearConfig::new(self.hidden_size, q_out)
                    .with_bias(self.attention_bias)
                    .init(device),
                k_proj: LinearConfig::new(self.hidden_size, kv_out)
                    .with_bias(self.attention_bias)
                    .init(device),
                v_proj: LinearConfig::new(self.hidden_size, kv_out)
                    .with_bias(self.attention_bias)
                    .init(device),
                o_proj: LinearConfig::new(q_out, self.hidden_size)
                    .with_bias(self.attention_bias)
                    .init(device),
                q_norm: RmsNormConfig::new(self.head_dim)
                    .with_epsilon(self.rms_norm_eps)
                    .init(device),
                k_norm: RmsNormConfig::new(self.head_dim)
                    .with_epsilon(self.rms_norm_eps)
                    .init(device),
            },
            mlp: Qwen3TtsTextMlp {
                gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(false)
                    .init(device),
                down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(false)
                    .init(device),
            },
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
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
