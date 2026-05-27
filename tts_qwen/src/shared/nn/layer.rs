use burn::module::Module;
use burn::nn::RmsNorm;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::attention::{AttentionPosition, Qwen3TtsAttention};
use super::mlp::Qwen3TtsTextMlp;
use super::rms_norm::qwen_rms_norm;
use crate::shared::runtime::cache::KeyValueCache;

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
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = qwen_rms_norm(&self.input_layernorm, x);
        let attn_out =
            self.self_attn
                .forward(x, num_heads, num_kv_heads, head_dim, position, mask, cache);
        let x = residual + attn_out;

        let residual = x.clone();
        let x = qwen_rms_norm(&self.post_attention_layernorm, x);
        residual + self.mlp.forward(x)
    }
}
