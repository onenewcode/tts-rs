use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::attention::AttentionPosition;
use super::kv::KeyValueCache;
use super::layer::Qwen3TtsDecoderLayer;
use super::mlp::Qwen3TtsTalkerResizeMlp;
use super::predictor::Qwen3TtsTalkerCodePredictor;
use super::rope::Qwen3RotaryEncoding;

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerModelBundle<B: Backend> {
    pub talker: Qwen3TtsTalker<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalker<B: Backend> {
    pub model: Qwen3TtsTalkerModel<B>,
    pub text_projection: Qwen3TtsTalkerResizeMlp<B>,
    pub codec_head: Linear<B>,
    pub code_predictor: Qwen3TtsTalkerCodePredictor<B>,
    pub mrope: Qwen3RotaryEncoding<B>,
}

impl<B> Qwen3TtsTalker<B>
where
    B: Backend,
{
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        position_ids: Tensor<B, 3, Int>,
        mask: Option<Tensor<B, 2, Int>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let causal_mask = (seq_len == key_len).then(|| {
            generate_autoregressive_mask::<B>(batch_size, seq_len, &device).unsqueeze_dim::<4>(1)
        });
        let padding_mask =
            mask.map(|mask| mask.equal_elem(0).unsqueeze::<4>().repeat_dim(2, seq_len));
        let final_mask = match (causal_mask, padding_mask) {
            (Some(causal), Some(padding)) => Some(causal.bool_or(padding)),
            (Some(causal), None) => Some(causal),
            (None, Some(padding)) => Some(padding),
            (None, None) => None,
        };

        self.model.decode(
            inputs_embeds,
            position_ids,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.mrope,
            final_mask.as_ref(),
            cache,
        )
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerModel<B: Backend> {
    pub codec_embedding: Embedding<B>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub text_embedding: Embedding<B>,
}

impl<B> Qwen3TtsTalkerModel<B>
where
    B: Backend,
{
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        inputs_embeds: Tensor<B, 3>,
        position_ids: Tensor<B, 3, Int>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mrope: &Qwen3RotaryEncoding<B>,
        mask: Option<&Tensor<B, 4, Bool>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let (cos, sin) = mrope.get_cos_sin(position_ids);

        let mut x = inputs_embeds;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            x = layer.forward(
                x,
                num_heads,
                num_kv_heads,
                head_dim,
                AttentionPosition::Mrope {
                    cos: &cos,
                    sin: &sin,
                },
                mask,
                c,
            );
        }
        self.norm.forward(x)
    }
}
