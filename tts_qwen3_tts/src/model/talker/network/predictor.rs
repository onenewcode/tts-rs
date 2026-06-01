use burn::module::Module;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::attention::AttentionPosition;
use super::kv::KeyValueCache;
use super::layer::Qwen3TtsDecoderLayer;
use super::mask::build_attention_mask;
use super::rope::Qwen3StandardRotaryEncoding;

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictor<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
    pub rope: Qwen3StandardRotaryEncoding<B>,
}

impl<B> Qwen3TtsTalkerCodePredictor<B>
where
    B: Backend,
{
    // TODO 不允许有  #[allow(dead_code)]你要删除无用的代码
    #[allow(dead_code)]
    pub fn predict(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mask: Option<Tensor<B, 2, Int>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let final_mask = build_attention_mask(batch_size, seq_len, key_len, mask, &device);

        let x = if let Some(projection) = &self.small_to_mtp_projection {
            projection.forward(inputs_embeds)
        } else {
            inputs_embeds
        };

        let hidden_states = self.model.forward(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.rope,
            final_mask,
            cache,
        );

        let [batch_size, _seq_len, hidden_size] = hidden_states.dims();
        let mut all_logits = Vec::with_capacity(self.lm_head.len());
        for (i, head) in self.lm_head.iter().enumerate() {
            let group_hidden = hidden_states
                .clone()
                .slice([0..batch_size, i + 1..i + 2, 0..hidden_size])
                .reshape([batch_size, hidden_size]);
            all_logits.push(
                head.forward(group_hidden.cast(head.weight.val().dtype()))
                    .unsqueeze::<3>(),
            );
        }
        Tensor::cat(all_logits, 1)
    }
}

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
        let (cos, sin) = rope.get_cos_sin(
            batch_size,
            seq_len,
            start,
            inputs_embeds.dtype(),
            &inputs_embeds.device(),
        );
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
