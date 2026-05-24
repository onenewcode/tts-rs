use burn::module::Module;
use burn::nn::{RmsNorm, RotaryEncoding};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::attention::Qwen3TtsAttention;
use super::mlp::Qwen3TtsTextMlp;
use super::super::cache::KeyValueCache;

#[derive(Module, Debug)]
pub struct Qwen3TtsDecoderLayer<B: Backend> {
    pub self_attn: Qwen3TtsAttention<B>,
    pub mlp: Qwen3TtsTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3TtsDecoderLayer<B> {
    /// Forward pass with multimodal RoPE (pre-calculated cos/sin)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        mask: Tensor<B, 4, Bool>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward_mrope(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            cos,
            sin,
            mask,
            cache,
        );
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }

    /// Forward pass with standard RoPE (official Module)
    pub fn forward_with_rope(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Tensor<B, 4, Bool>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            mask,
            cache,
        );
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}
