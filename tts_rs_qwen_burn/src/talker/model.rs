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
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use super::nn::attention::{AttentionPosition, AttentionValueMode};
use super::nn::mlp::native_linear_3d;
use super::nn::rms_norm::qwen_rms_norm;
use super::nn::{
    Qwen3RotaryEncoding, Qwen3StandardRotaryEncoding, Qwen3TtsDecoderLayer, Qwen3TtsTalkerResizeMlp,
};
use crate::shared::runtime::cache::KeyValueCache;
use crate::talker::types::{TalkerActivations, TalkerAttentionActivations};

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
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        TalkerActivations<B>,
        TalkerAttentionActivations<B>,
    ) {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let key_len = cache.first().map_or(seq_len, |cache| cache.len() + seq_len);
        let device = inputs_embeds.device();
        let final_mask = build_attention_mask(batch_size, seq_len, key_len, mask, &device);

        let (hidden_states, mut activations, attention_activations) = self.model.forward(
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
        (hidden_states, logits, activations, attention_activations)
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
    ) -> (
        Tensor<B, 3>,
        TalkerActivations<B>,
        TalkerAttentionActivations<B>,
    ) {
        let (cos, sin) = mrope.get_cos_sin(position_ids, inputs_embeds.dtype());

        let mut x = inputs_embeds;
        let mut activations = BTreeMap::new();
        let mut attention_activations = BTreeMap::new();
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
                AttentionValueMode::CastSoftmaxToModelDTypeBeforeValueMatmul,
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
                attention_activations.insert(
                    format!("layers.{idx}.attn.weights"),
                    output
                        .attn_weights
                        .expect("attention weights collected when requested"),
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
                activations.insert(
                    format!("layers.{idx}.q_norm.output"),
                    output.q_norm.expect("q_norm collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.k_norm.output"),
                    output.k_norm.expect("k_norm collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.q_rot.output"),
                    output.q_rot.expect("q_rot collected when requested"),
                );
                activations.insert(
                    format!("layers.{idx}.k_rot.output"),
                    output.k_rot.expect("k_rot collected when requested"),
                );
            }
            x = output.hidden;
        }

        let x = qwen_rms_norm(&self.norm, x);
        if collect_activations {
            activations.insert("model.norm.output".to_string(), x.clone());
        }
        (x, activations, attention_activations)
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
            native_linear_3d(
                projection,
                inputs_embeds.cast(projection.weight.val().dtype()),
            )
        } else {
            inputs_embeds
        };
        let x = x.cast(self.model.layers[0].self_attn.q_proj.weight.val().dtype());

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
        let key_len = start + seq_len;
        let (cos, sin) = rope.get_cos_sin(
            batch_size,
            seq_len,
            start,
            inputs_embeds.dtype(),
            &inputs_embeds.device(),
        );
        let mut x = inputs_embeds;
        for (layer_idx, (layer, c)) in self.layers.iter().zip(cache.iter_mut()).enumerate() {
            x = layer
                .forward(
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
                    code_predictor_attention_value_mode(key_len, layer_idx),
                    false,
                )
                .hidden;
        }
        qwen_rms_norm(&self.norm, x)
    }

    pub fn forward_with_activations(
        &self,
        inputs_embeds: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &Qwen3StandardRotaryEncoding<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut [KeyValueCache<B>],
    ) -> (
        Tensor<B, 3>,
        TalkerActivations<B>,
        TalkerAttentionActivations<B>,
    ) {
        let [batch_size, seq_len, _] = inputs_embeds.dims();
        let start = cache.first().map_or(0, KeyValueCache::len);
        let key_len = start + seq_len;
        let (cos, sin) = rope.get_cos_sin(
            batch_size,
            seq_len,
            start,
            inputs_embeds.dtype(),
            &inputs_embeds.device(),
        );
        let mut x = inputs_embeds;
        let mut activations = BTreeMap::new();
        let mut attention_activations = BTreeMap::new();
        for (idx, (layer, c)) in self.layers.iter().zip(cache.iter_mut()).enumerate() {
            let output = layer.forward(
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
                code_predictor_attention_value_mode(key_len, idx),
                true,
            );
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
            attention_activations.insert(
                format!("layers.{idx}.attn.weights"),
                output
                    .attn_weights
                    .expect("attention weights collected when requested"),
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
            activations.insert(
                format!("layers.{idx}.q_norm.output"),
                output.q_norm.expect("q_norm collected when requested"),
            );
            activations.insert(
                format!("layers.{idx}.k_norm.output"),
                output.k_norm.expect("k_norm collected when requested"),
            );
            activations.insert(
                format!("layers.{idx}.q_rot.output"),
                output.q_rot.expect("q_rot collected when requested"),
            );
            activations.insert(
                format!("layers.{idx}.k_rot.output"),
                output.k_rot.expect("k_rot collected when requested"),
            );
            x = output.hidden;
        }
        let x = qwen_rms_norm(&self.norm, x);
        activations.insert("model.norm.output".to_string(), x.clone());
        (x, activations, attention_activations)
    }
}

fn code_predictor_attention_value_mode(key_len: usize, layer_idx: usize) -> AttentionValueMode {
    if let Ok(schedule) = std::env::var("QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_SCHEDULE") {
        if schedule_matches(&schedule, key_len, layer_idx) {
            return AttentionValueMode::PyTorchEagerBf16ScoresAndValueMatmul;
        }
        return code_predictor_attention_default_mode();
    }

    if let Ok(value) = std::env::var("QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN") {
        let layer_matches = std::env::var("QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER")
            .map(|layers| {
                layers
                    .split(',')
                    .filter_map(|item| item.trim().parse::<usize>().ok())
                    .any(|allowed| allowed == layer_idx)
            })
            .unwrap_or(true);
        if value
            .split(',')
            .filter_map(|item| item.trim().parse::<usize>().ok())
            .any(|allowed| allowed == key_len)
            && layer_matches
        {
            return AttentionValueMode::PyTorchEagerBf16ScoresAndValueMatmul;
        }
    }

    code_predictor_attention_default_mode()
}

fn code_predictor_attention_default_mode() -> AttentionValueMode {
    match std::env::var("QWEN_TTS_CODE_PREDICTOR_ATTENTION").as_deref() {
        Ok("eager") => AttentionValueMode::EagerModelDTypeScoresAndValueMatmul,
        Ok("keep_f32") => AttentionValueMode::KeepSoftmaxF32ForValueMatmul,
        Ok("pytorch_bf16") => AttentionValueMode::PyTorchEagerBf16ScoresAndValueMatmul,
        _ => AttentionValueMode::CastSoftmaxToModelDTypeBeforeValueMatmul,
    }
}

fn schedule_matches(schedule: &str, key_len: usize, layer_idx: usize) -> bool {
    schedule.split(';').any(|entry| {
        let Some((keys, layers)) = entry.split_once(':') else {
            return false;
        };
        key_spec_matches(keys.trim(), key_len) && list_contains(layers, layer_idx)
    })
}

fn key_spec_matches(spec: &str, key_len: usize) -> bool {
    spec.split('|').any(|part| {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let Some(start) = start.trim().parse::<usize>().ok() else {
                return false;
            };
            let Some(end) = end.trim().parse::<usize>().ok() else {
                return false;
            };
            return (start..=end).contains(&key_len);
        }
        part.parse::<usize>() == Ok(key_len)
    })
}

fn list_contains(list: &str, value: usize) -> bool {
    list.split(',')
        .filter_map(|item| item.trim().parse::<usize>().ok())
        .any(|allowed| allowed == value)
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
