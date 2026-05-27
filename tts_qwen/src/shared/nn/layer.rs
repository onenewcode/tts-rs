use burn::module::Module;
use burn::nn::RmsNorm;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::attention::{AttentionPosition, AttentionValueMode, Qwen3TtsAttention};
use super::mlp::Qwen3TtsTextMlp;
use super::rms_norm::qwen_rms_norm;
use crate::shared::runtime::cache::KeyValueCache;

pub struct DecoderLayerOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub input_norm: Option<Tensor<B, 3>>,
    pub attn_residual: Option<Tensor<B, 3>>,
    pub post_attention_norm: Option<Tensor<B, 3>>,
    pub mlp_gate: Option<Tensor<B, 3>>,
    pub mlp_up: Option<Tensor<B, 3>>,
    pub mlp_activated_gate: Option<Tensor<B, 3>>,
    pub mlp_product: Option<Tensor<B, 3>>,
    pub attn_output: Option<Tensor<B, 3>>,
    pub attn_weights: Option<Tensor<B, 4>>,
    pub mlp_output: Option<Tensor<B, 3>>,
    pub q_proj: Option<Tensor<B, 3>>,
    pub k_proj: Option<Tensor<B, 3>>,
    pub v_proj: Option<Tensor<B, 3>>,
    pub q_norm: Option<Tensor<B, 3>>,
    pub k_norm: Option<Tensor<B, 3>>,
    pub q_rot: Option<Tensor<B, 3>>,
    pub k_rot: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsDecoderLayer<B: Backend> {
    pub self_attn: Qwen3TtsAttention<B>,
    pub mlp: Qwen3TtsTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B> Qwen3TtsDecoderLayer<B>
where
    B: Backend,
{
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
        attention_value_mode: AttentionValueMode,
        collect_activations: bool,
    ) -> DecoderLayerOutput<B> {
        let residual = x.clone();
        let x = qwen_rms_norm(&self.input_layernorm, x);
        let input_norm = x.clone();
        let attn_debug = self.self_attn.forward_debug(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            position,
            mask,
            cache,
            attention_value_mode,
        );
        let attn_out = attn_debug.output;
        let x = residual + attn_out.clone();
        let attn_residual = x.clone();

        let residual = x.clone();
        let x = qwen_rms_norm(&self.post_attention_layernorm, x);
        let post_attention_norm = x.clone();
        let mlp_output = self.mlp.forward_with_activations(x);
        let hidden = residual + mlp_output.output.clone();

        DecoderLayerOutput {
            hidden,
            input_norm: collect_activations.then_some(input_norm),
            attn_residual: collect_activations.then_some(attn_residual),
            post_attention_norm: collect_activations.then_some(post_attention_norm),
            mlp_gate: collect_activations.then_some(mlp_output.gate),
            mlp_up: collect_activations.then_some(mlp_output.up),
            mlp_activated_gate: collect_activations.then_some(mlp_output.activated_gate),
            mlp_product: collect_activations.then_some(mlp_output.product),
            attn_output: collect_activations.then_some(attn_out),
            attn_weights: collect_activations.then_some(attn_debug.attn_weights),
            mlp_output: collect_activations.then_some(mlp_output.output),
            q_proj: collect_activations.then_some(attn_debug.q_proj),
            k_proj: collect_activations.then_some(attn_debug.k_proj),
            v_proj: collect_activations.then_some(attn_debug.v_proj),
            q_norm: collect_activations.then_some(attn_debug.q_norm),
            k_norm: collect_activations.then_some(attn_debug.k_norm),
            q_rot: collect_activations.then_some(attn_debug.q_rot),
            k_rot: collect_activations.then_some(attn_debug.k_rot),
        }
    }
}
