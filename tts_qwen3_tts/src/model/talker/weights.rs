use std::path::Path;

use burn::nn::{EmbeddingConfig, LinearConfig, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn_store::KeyRemapper;
use burn_store::{ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use super::attention::Qwen3TtsAttention;
use super::layer::Qwen3TtsDecoderLayer;
use super::mlp::{Qwen3TtsTalkerResizeMlp, Qwen3TtsTextMlp};
use super::rope::{Qwen3RotaryEncoding, Qwen3StandardRotaryEncoding};
use crate::model::talker::config::{Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig};
use crate::model::talker::network::{
    Qwen3TtsTalker, Qwen3TtsTalkerCodePredictor, Qwen3TtsTalkerCodePredictorModel,
    Qwen3TtsTalkerModel, Qwen3TtsTalkerModelBundle,
};
use crate::Qwen3TtsLoadError;

const TALKER_LOAD_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.weight$", "${1}norm.gamma")];
#[cfg(test)]
const TALKER_EXPORT_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.gamma$", "${1}norm.weight")];

fn talker_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(TALKER_LOAD_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

#[cfg(test)]
fn talker_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(TALKER_EXPORT_KEY_PATTERNS.to_vec())
        .expect("static regex remapping must compile")
}

#[derive(Debug)]
pub struct LoadedQwen3TtsTalker<B: Backend> {
    pub config: Qwen3TtsTalkerConfig,
    pub model: Qwen3TtsTalkerModelBundle<B>,
}

pub fn load_qwen3_tts_talker_for_inference<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    load_qwen3_tts_talker_with_adapter::<B, _>(
        config_path,
        weights_path,
        device,
        PyTorchToBurnAdapter,
    )
}

fn load_qwen3_tts_talker_with_adapter<B: Backend, A: ModuleAdapter + 'static>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
    adapter: A,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts talker"
    );
    let config = Qwen3TtsTalkerConfig::load_from_path(&config_path)?;
    let mut model = config.init_model_bundle(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(adapter)
        .remap(talker_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    let applied = apply_result.applied.len();
    let skipped = apply_result.skipped.len();
    let missing = apply_result.missing.len();
    let unused = apply_result.unused.len();
    if unused != 0 {
        tracing::warn!(
            unused,
            "qwen3 tts talker weights left tensors unused during load"
        );
    }
    tracing::info!(
        applied,
        skipped,
        missing,
        unused,
        "loaded qwen3 tts talker weights"
    );

    Ok(LoadedQwen3TtsTalker { config, model })
}

impl Qwen3TtsTalkerConfig {
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

#[cfg(test)]
mod tests {
    use super::{talker_export_key_remapper, talker_load_key_remapper};

    #[test]
    fn talker_remappers_compile() {
        let _ = talker_load_key_remapper();
        let _ = talker_export_key_remapper();
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
