//! # Talker Model Architecture
//!
//! ```text
//! Qwen3TtsCheckpoint
//!   └── Qwen3TtsTalker
//!         ├── Qwen3TtsTalkerModel (28+ decoder layers)
//!         │     ├── text_embedding
//!         │     ├── codec_embedding
//!         │     ├── layers: Vec<Qwen3TtsDecoderLayer>
//!         │     └── norm: RmsNorm
//!         ├── text_projection: ResizeMlp
//!         ├── codec_head: Linear → logits
//!         ├── code_predictor: separate decoder for codec groups
//!         └── mrope: Qwen3RotaryEncoding (multimodal RoPE)
//! ```

use std::collections::BTreeMap;

use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Embedding, Linear, RmsNorm, RotaryEncoding};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::cache::KeyValueCache;
use super::nn::attention::AttentionPosition;
use super::nn::mlp::native_linear_3d;
use super::nn::rms_norm::qwen_rms_norm;
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

impl<B> Qwen3TtsTalker<B>
where
    B: Backend,
{
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        position_ids: Tensor<B, 3, Int>,
        mask: Option<Tensor<B, 2, Int>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cache: &mut [KeyValueCache<B>],
        collect_activations: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, BTreeMap<String, Tensor<B, 3>>) {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let final_mask = build_attention_mask(batch_size, seq_len, key_len, mask, &device);

        let (hidden_states, mut activations) = self.model.forward(
            inputs_embeds,
            position_ids,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.mrope,
            final_mask,
            cache,
            collect_activations,
        );
        let logits = native_linear_3d(&self.codec_head, hidden_states.clone());
        if collect_activations {
            activations.insert("codec_head.logits".to_string(), logits.clone());
        }
        (hidden_states, logits, activations)
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
        collect_activations: bool,
    ) -> (Tensor<B, 3>, BTreeMap<String, Tensor<B, 3>>) {
        let (cos, sin) = mrope.get_cos_sin(position_ids, inputs_embeds.dtype());

        let mut x = inputs_embeds;
        let mut activations = BTreeMap::new();
        for (idx, (layer, c)) in self.layers.iter().zip(cache.iter_mut()).enumerate() {
            let output = layer.forward(
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
                collect_activations,
            );
            if collect_activations {
                activations.insert(
                    format!("layers.{idx}.input_norm.output"),
                    output
                        .input_norm
                        .expect("input norm output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.attn.output"),
                    output
                        .attn_output
                        .expect("attention output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.attn_residual.output"),
                    output
                        .attn_residual
                        .expect("attention residual collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.post_attention_norm.output"),
                    output
                        .post_attention_norm
                        .expect("post attention norm output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.mlp.gate"),
                    output
                        .mlp_gate
                        .expect("mlp gate output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.mlp.up"),
                    output
                        .mlp_up
                        .expect("mlp up output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.mlp.activated_gate"),
                    output
                        .mlp_activated_gate
                        .expect("mlp activated gate output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.mlp.product"),
                    output
                        .mlp_product
                        .expect("mlp product output collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.mlp.output"),
                    output
                        .mlp_output
                        .expect("mlp output collected when requested"),
                );
                activations.insert(format!("layers.{idx}.hidden.output"), output.hidden.clone());
                activations.insert(
                    format!("layers.{idx}.q_proj.output"),
                    output.q_proj.expect("q_proj collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.k_proj.output"),
                    output.k_proj.expect("k_proj collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.v_proj.output"),
                    output.v_proj.expect("v_proj collected when requested"),
                );
            }
            x = output.hidden;
        }

        let x = qwen_rms_norm(&self.norm, x);
        if collect_activations {
            activations.insert("model.norm.output".to_string(), x.clone());
        }
        (x, activations)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictor<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
    pub rope: RotaryEncoding<B>,
}

impl<B> Qwen3TtsTalkerCodePredictor<B>
where
    B: Backend,
{
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
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let final_mask = build_attention_mask(batch_size, seq_len, key_len, mask, &device);

        let x = if let Some(projection) = &self.small_to_mtp_projection {
            native_linear_3d(projection, inputs_embeds)
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

impl<B> Qwen3TtsTalkerCodePredictorModel<B>
where
    B: Backend,
{
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut [KeyValueCache<B>],
    ) -> Tensor<B, 3> {
        let mut x = inputs_embeds;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            x = layer
                .forward(
                    x,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    AttentionPosition::Standard { rope },
                    mask.clone(),
                    c,
                    false,
                )
                .hidden;
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
