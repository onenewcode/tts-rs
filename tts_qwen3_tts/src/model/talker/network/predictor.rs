use burn::module::Module;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::attention::AttentionPosition;
use super::kv::KeyValueCache;
use super::layer::Qwen3TtsDecoderLayer;
use super::rope::Qwen3StandardRotaryEncoding;

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictor<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
    pub rope: Qwen3StandardRotaryEncoding<B>,
}

impl<B> Qwen3TtsTalkerCodePredictor<B> where B: Backend {}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictorModel<B: Backend> {
    pub codec_embedding: Vec<Embedding<B>>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B> Qwen3TtsTalkerCodePredictorModel<B>
where
    B: Backend,
{
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &Qwen3StandardRotaryEncoding<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let start = cache.first().map_or(0, KeyValueCache::len);
        let (cos, sin) = rope.get_cos_sin(batch_size, seq_len, start, &inputs_embeds.device());
        let mut x = inputs_embeds;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            x = layer.forward(
                x,
                num_heads,
                num_kv_heads,
                head_dim,
                AttentionPosition::Standard {
                    cos: cos.clone(),
                    sin: sin.clone(),
                },
                mask.clone(),
                c,
            );
        }
        self.norm.forward(x)
    }
}
