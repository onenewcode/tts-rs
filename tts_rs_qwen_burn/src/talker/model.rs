use burn::module::Module;
use burn::nn::{Embedding, Linear, RmsNorm, RotaryEncoding};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::cache::KeyValueCache;
use super::nn::{Qwen3RotaryEncoding, Qwen3TtsDecoderLayer, Qwen3TtsTalkerResizeMlp};

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

impl<B: Backend> Qwen3TtsTalker<B> {
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
        let device = inputs_embeds.device();

        let causal_mask = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &device)
            .unsqueeze::<4>()
            .repeat_dim(0, batch_size);

        let final_mask = if let Some(padding_mask) = mask {
            let padding_mask = padding_mask
                .equal_elem(0)
                .unsqueeze::<4>()
                .repeat_dim(2, seq_len);
            causal_mask.bool_or(padding_mask)
        } else {
            causal_mask
        };

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
        let logits = self.codec_head.forward(hidden_states.clone());
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

impl<B: Backend> Qwen3TtsTalkerModel<B> {
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        position_ids: Tensor<B, 3, Int>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mrope: &Qwen3RotaryEncoding<B>,
        mask: Tensor<B, 4, Bool>,
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
                cos.clone(),
                sin.clone(),
                mask.clone(),
                c,
            );
        }
        self.norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictor<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
    pub rope: RotaryEncoding<B>,
}

impl<B: Backend> Qwen3TtsTalkerCodePredictor<B> {
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mask: Option<Tensor<B, 2, Int>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let device = inputs_embeds.device();

        let causal_mask = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &device)
            .unsqueeze::<4>()
            .repeat_dim(0, batch_size);

        let final_mask = if let Some(padding_mask) = mask {
            let padding_mask = padding_mask
                .equal_elem(0)
                .unsqueeze::<4>()
                .repeat_dim(2, seq_len);
            causal_mask.bool_or(padding_mask)
        } else {
            causal_mask
        };

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

        // Standard LM head application
        let [batch_size, _seq_len, hidden_size] = hidden_states.dims();
        let mut all_logits = Vec::with_capacity(self.lm_head.len());
        for (i, head) in self.lm_head.iter().enumerate() {
            let group_hidden = hidden_states
                .clone()
                .slice([0..batch_size, i + 1..i + 2, 0..hidden_size])
                .reshape([batch_size, hidden_size]);
            all_logits.push(head.forward(group_hidden).unsqueeze::<3>());
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

impl<B: Backend> Qwen3TtsTalkerCodePredictorModel<B> {
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Tensor<B, 4, Bool>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let mut x = inputs_embeds;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            x = layer.forward_with_rope(
                x,
                num_heads,
                num_kv_heads,
                head_dim,
                rope,
                mask.clone(),
                c,
            );
        }
        self.norm.forward(x)
    }
}
