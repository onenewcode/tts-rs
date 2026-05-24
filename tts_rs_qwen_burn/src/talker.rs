use std::path::{Path, PathBuf};

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::manifest::{
    LoadReport, VerificationArtifacts, WeightVerificationReport, verify_module_weights,
};
use crate::{Qwen3TtsLoadError, Qwen3TtsVerifyError};

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

#[derive(Module, Debug)]
pub struct Qwen3TtsCheckpoint<B: Backend> {
    pub talker: Qwen3TtsTalkerForConditionalGeneration<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerForConditionalGeneration<B: Backend> {
    pub model: Qwen3TtsTalkerModel<B>,
    pub text_projection: Qwen3TtsTalkerResizeMlp<B>,
    pub codec_head: Linear<B>,
    pub code_predictor: Qwen3TtsTalkerCodePredictorForConditionalGeneration<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerModel<B: Backend> {
    pub codec_embedding: Embedding<B>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub text_embedding: Embedding<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictorForConditionalGeneration<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictorModel<B: Backend> {
    pub codec_embedding: Vec<Embedding<B>>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsDecoderLayer<B: Backend> {
    pub self_attn: Qwen3TtsAttention<B>,
    pub mlp: Qwen3TtsTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTextMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerResizeMlp<B: Backend> {
    pub linear_fc1: Linear<B>,
    pub linear_fc2: Linear<B>,
}

#[derive(Debug)]
pub struct LoadedQwen3TtsTalker<B: Backend> {
    pub config: Qwen3TtsConfig,
    pub model: Qwen3TtsCheckpoint<B>,
    pub load_report: LoadReport,
    pub model_dir: PathBuf,
    pub weights_path: PathBuf,
}

impl Qwen3TtsConfig {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = model_dir.as_ref().join("config.json");
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }

    pub fn init_checkpoint<B: Backend>(&self, device: &B::Device) -> Qwen3TtsCheckpoint<B> {
        Qwen3TtsCheckpoint {
            talker: self.talker_config.init(device),
        }
    }
}

impl Qwen3TtsTalkerConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Qwen3TtsTalkerForConditionalGeneration<B> {
        Qwen3TtsTalkerForConditionalGeneration {
            model: self.init_model(device),
            text_projection: self.init_text_projection(device),
            codec_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
            code_predictor: self.init_code_predictor(device),
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
    ) -> Qwen3TtsTalkerCodePredictorForConditionalGeneration<B> {
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

        Qwen3TtsTalkerCodePredictorForConditionalGeneration {
            model: code_predictor.init_model(self.hidden_size, device),
            lm_head: (0..num_heads)
                .map(|_| {
                    LinearConfig::new(code_predictor.hidden_size, code_predictor.vocab_size)
                        .with_bias(false)
                        .init(device)
                })
                .collect(),
            small_to_mtp_projection: projection,
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

pub fn load_qwen3_tts_talker<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let weights_path = model_dir.join("model.safetensors");
    let config = Qwen3TtsConfig::load_from_model_dir(&model_dir)?;
    let mut model = config.init_checkpoint(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(talker_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    if !apply_result.unused.is_empty() {
        return Err(Qwen3TtsLoadError::UnusedTensors {
            unused: apply_result.unused.len(),
        });
    }

    let load_report = LoadReport {
        applied: apply_result.applied.len(),
        skipped: apply_result.skipped.len(),
        missing: apply_result.missing.len(),
        unused: apply_result.unused.len(),
    };

    Ok(LoadedQwen3TtsTalker {
        config,
        model,
        load_report,
        model_dir,
        weights_path,
    })
}

pub fn verify_qwen3_tts_talker_weights<B: Backend>(
    model: &Qwen3TtsCheckpoint<B>,
    weights_path: impl AsRef<Path>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    verify_module_weights(
        model,
        weights_path,
        Some(talker_export_key_remapper()),
        artifacts,
    )
}

fn talker_load_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![(r"(.*)norm\.weight$", "${1}norm.gamma")])
        .expect("static regex remapping must compile")
}

fn talker_export_key_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![(r"(.*)norm\.gamma$", "${1}norm.weight")])
        .expect("static regex remapping must compile")
}

#[cfg(test)]
mod tests {
    use crate::{VerificationArtifacts, default_workspace_root, find_local_qwen_tts_model_dir};

    use super::*;

    type TestBackend = burn::backend::Flex;

    #[test]
    fn real_checkpoint_talker_weights_roundtrip() {
        let workspace_root = default_workspace_root();
        let model_dir =
            find_local_qwen_tts_model_dir(&workspace_root).expect("local qwen model directory");
        let device = Default::default();

        let loaded = load_qwen3_tts_talker::<TestBackend>(&model_dir, &device)
            .expect("checkpoint should load");
        assert_eq!(loaded.load_report.missing, 0);
        assert_eq!(loaded.load_report.unused, 0);
        assert_eq!(loaded.load_report.skipped, 0);

        let artifacts = VerificationArtifacts::new(
            workspace_root.join("artifacts/qwen3_tts/talker/test_roundtrip"),
        );
        let verification =
            verify_qwen3_tts_talker_weights(&loaded.model, &loaded.weights_path, Some(&artifacts))
                .expect("loaded model should roundtrip back to the original checkpoint");
        assert_eq!(verification.tensor_count, loaded.load_report.applied);
        assert_eq!(verification.tensor_count, 402);
    }
}
