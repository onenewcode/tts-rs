use burn::nn::{EmbeddingConfig, LinearConfig, RmsNormConfig, RotaryEncodingConfig};
use burn::tensor::backend::Backend;

use super::config::{Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};
use super::model::{
    Qwen3TtsCheckpoint, Qwen3TtsTalker, Qwen3TtsTalkerCodePredictor,
    Qwen3TtsTalkerCodePredictorModel, Qwen3TtsTalkerModel,
};
use super::nn::{
    Qwen3RotaryEncoding, Qwen3TtsAttention, Qwen3TtsDecoderLayer, Qwen3TtsTalkerResizeMlp,
    Qwen3TtsTextMlp,
};

impl Qwen3TtsConfig {
    pub fn init_checkpoint<B: Backend>(&self, device: &B::Device) -> Qwen3TtsCheckpoint<B> {
        Qwen3TtsCheckpoint {
            talker: self.talker_config.init(device),
        }
    }
}

impl Qwen3TtsTalkerConfig {
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
            rope: RotaryEncodingConfig::new(
                code_predictor.max_position_embeddings,
                code_predictor.head_dim,
            )
            .with_theta(code_predictor.rope_theta as f32)
            .init(device),
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
