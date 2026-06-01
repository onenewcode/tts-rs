use burn::module::Module;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::attention::AttentionPosition;
use super::kv::KeyValueCache;
use super::layer::Qwen3TtsDecoderLayer;
use super::mask::build_attention_mask;
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
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let final_mask = build_attention_mask(batch_size, seq_len, key_len, mask, &device);

        let hidden_states = self.model.forward(
            inputs_embeds,
            position_ids,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.mrope,
            final_mask,
            cache,
        );
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let logits = self.codec_head.forward(
            hidden_states
                .clone()
                .reshape([batch_size * seq_len, hidden_size])
                .cast(self.codec_head.weight.val().dtype()),
        );
        let logits_vocab = logits.dims()[1];
        let logits = logits.reshape([batch_size, seq_len, logits_vocab]);
        let [_logits_batch, _logits_seq, logits_vocab] = logits.dims();
        debug_assert_eq!(hidden_size, self.codec_head.weight.dims()[0]);
        debug_assert_eq!(logits_vocab, self.codec_head.weight.dims()[1]);
        (hidden_states, logits)
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
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        position_ids: Tensor<B, 3, Int>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mrope: &Qwen3RotaryEncoding<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let (cos, sin) = mrope.get_cos_sin(position_ids, inputs_embeds.dtype());

        let mut x = inputs_embeds;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            x = layer.forward(
                x,
                num_heads,
                num_kv_heads,
                head_dim,
                AttentionPosition::Mrope {
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
