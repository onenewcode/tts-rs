use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::kernels::attention::AttentionPosition;
use crate::kernels::layer::Qwen3TtsDecoderLayer;
use crate::kernels::mlp::Qwen3TtsTalkerResizeMlp;
use crate::kernels::mlp::native_linear_3d;
use crate::kernels::norm::qwen_rms_norm;
use crate::kernels::rope::{Qwen3RotaryEncoding, Qwen3StandardRotaryEncoding};
use crate::runtime::kv::KeyValueCache;

#[derive(Module, Debug)]
pub struct Qwen3TtsCheckpoint<B: Backend> {
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
    pub fn infer(
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

        let hidden_states = self.model.run_layers(
            inputs_embeds,
            position_ids,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.mrope,
            final_mask,
            cache,
        );
        let logits = native_linear_3d(&self.codec_head, hidden_states.clone());
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
    pub fn run_layers(
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

        qwen_rms_norm(&self.norm, x)
    }
}

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
            native_linear_3d(
                projection,
                inputs_embeds.cast(projection.weight.val().dtype()),
            )
        } else {
            inputs_embeds
        };
        let x = x.cast(self.model.layers[0].self_attn.q_proj.weight.val().dtype());

        let hidden_states = self.model.run_layers(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.rope,
            final_mask,
            cache,
        );

        // Standard LM head application
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
    pub fn run_layers(
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
        qwen_rms_norm(&self.norm, x)
    }
}

pub(crate) fn build_attention_mask<B: Backend>(
    batch_size: usize,
    query_len: usize,
    key_len: usize,
    padding_mask: Option<Tensor<B, 2, Int>>,
    device: &B::Device,
) -> Option<Tensor<B, 4, Bool>> {
    tracing::debug!(
        batch_size,
        query_len,
        key_len,
        has_padding_mask = padding_mask.is_some(),
        has_causal_mask = query_len == key_len,
        "building attention mask"
    );
    let causal_mask = (query_len == key_len).then(|| {
        generate_autoregressive_mask::<B>(batch_size, query_len, device).unsqueeze_dim::<4>(1)
    });

    let padding_mask =
        padding_mask.map(|mask| mask.equal_elem(0).unsqueeze::<4>().repeat_dim(2, query_len));

    match (causal_mask, padding_mask) {
        (Some(causal), Some(padding)) => Some(causal.bool_or(padding)),
        (Some(causal), None) => Some(causal),
        (None, Some(padding)) => Some(padding),
        (None, None) => None,
    }
}
