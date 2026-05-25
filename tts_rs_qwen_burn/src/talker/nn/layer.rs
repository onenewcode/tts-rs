use burn::module::Module;
use burn::nn::RmsNorm;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::super::cache::KeyValueCache;
use super::attention::{AttentionPosition, Qwen3TtsAttention};
use super::mlp::Qwen3TtsTextMlp;
use super::rms_norm::qwen_rms_norm;

pub struct DecoderLayerOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub input_norm: Option<Tensor<B, 3>>,
    pub attn_residual: Option<Tensor<B, 3>>,
    pub post_attention_norm: Option<Tensor<B, 3>>,
    pub attn_output: Option<Tensor<B, 3>>,
    pub mlp_output: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsDecoderLayer<B: Backend> {
    pub self_attn: Qwen3TtsAttention<B>,
    pub mlp: Qwen3TtsTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3TtsDecoderLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<'_, B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
        collect_activations: bool,
    ) -> DecoderLayerOutput<B> {
        let residual = x.clone();
        let x = qwen_rms_norm(&self.input_layernorm, x);
        let input_norm = x.clone();
        let attn_output =
            self.self_attn
                .forward(x, num_heads, num_kv_heads, head_dim, position, mask, cache);
        let x = residual + attn_output.clone();
        let attn_residual = x.clone();

        let residual = x.clone();
        let x = qwen_rms_norm(&self.post_attention_layernorm, x);
        let post_attention_norm = x.clone();
        let mlp_output = self.mlp.forward(x);
        let hidden = residual + mlp_output.clone();

        DecoderLayerOutput {
            hidden,
            input_norm: collect_activations.then_some(input_norm),
            attn_residual: collect_activations.then_some(attn_residual),
            post_attention_norm: collect_activations.then_some(post_attention_norm),
            attn_output: collect_activations.then_some(attn_output),
            mlp_output: collect_activations.then_some(mlp_output),
        }
    }
}
