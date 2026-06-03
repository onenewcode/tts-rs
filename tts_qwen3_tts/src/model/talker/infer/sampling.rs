//! Configurable token sampling strategies for autoregressive generation.
//!
//! Supports greedy argmax and randomized sampling with:
//! suppress -> temperature -> top-k -> top-p -> softmax -> categorical.

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, IndexingUpdateOp, Int, Tensor};

use super::select_last_sequence_step;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            repetition_penalty: None,
        }
    }
}

pub fn sample_token<B: Backend>(
    logits: Tensor<B, 3>,
    sampling: &SamplingConfig,
    suppress_token_ids: &[usize],
) -> Tensor<B, 2, Int> {
    let mut logits_2d = prepare_last_step_logits(logits, suppress_token_ids);

    if !sampling.do_sample {
        return logits_2d.argmax(1);
    }

    logits_2d = logits_2d.dequantize();
    let logits_dtype = logits_2d.dtype();
    if logits_dtype.size() < DType::F32.size() {
        logits_2d = logits_2d.cast(DType::F32);
    }

    let [batch_size, vocab_size] = logits_2d.dims();
    logits_2d = logits_2d.div_scalar(sampling.temperature.max(1e-5));

    if let Some(k) = sampling.top_k.filter(|k| *k > 0 && *k < vocab_size) {
        let kth_value = logits_2d
            .clone()
            .topk(k, 1)
            .slice([0..batch_size, k - 1..k]);
        let mask = logits_2d.clone().lower(kth_value);
        logits_2d = logits_2d.mask_fill(mask, f32::NEG_INFINITY);
    }

    if sampling.top_p.is_finite() && sampling.top_p > 0.0 && sampling.top_p < 1.0 {
        let (sorted_vals, sorted_idx) = logits_2d.clone().sort_descending_with_indices(1);
        let sorted_probs = softmax(sorted_vals, 1);
        let cumsum = sorted_probs.clone().cumsum(1);
        let sorted_keep: Tensor<B, 2, Bool> = cumsum.sub(sorted_probs).lower_elem(sampling.top_p);
        let inverse = sorted_idx.argsort(1);
        let orig_keep: Tensor<B, 2, Bool> = sorted_keep.int().gather(1, inverse).bool();
        logits_2d = logits_2d.mask_fill(orig_keep.bool_not(), f32::NEG_INFINITY);
    }

    let mut probs = softmax(logits_2d, 1);
    if probs.dtype() != logits_dtype {
        probs = probs.cast(logits_dtype);
    }
    probs.categorical(1)
}

fn prepare_last_step_logits<B: Backend>(
    logits: Tensor<B, 3>,
    suppress_token_ids: &[usize],
) -> Tensor<B, 2> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let logits_2d = if seq_len == 1 {
        logits.reshape([batch_size, vocab_size])
    } else {
        select_last_sequence_step(logits).reshape([batch_size, vocab_size])
    };
    apply_suppress_mask(logits_2d, suppress_token_ids)
}

fn apply_suppress_mask<B: Backend>(
    logits: Tensor<B, 2>,
    suppress_token_ids: &[usize],
) -> Tensor<B, 2> {
    if suppress_token_ids.is_empty() {
        return logits;
    }

    let [batch_size, vocab_size] = logits.dims();
    let device = logits.device();
    if let Some(suppress_mask) =
        suppress_token_mask::<B>(batch_size, vocab_size, suppress_token_ids, &device)
    {
        logits.mask_fill(suppress_mask, f32::NEG_INFINITY)
    } else {
        logits
    }
}

fn suppress_token_mask<B: Backend>(
    batch_size: usize,
    vocab_size: usize,
    suppress_token_ids: &[usize],
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    let valid_ids = suppress_token_ids
        .iter()
        .copied()
        .filter(|id| *id < vocab_size)
        .filter_map(|id| i64::try_from(id).ok())
        .collect::<Vec<_>>();
    if valid_ids.is_empty() {
        return None;
    }

    let suppress_len = valid_ids.len();
    let token_ids = Tensor::<B, 1, Int>::from_ints(valid_ids.as_slice(), device)
        .reshape([1, suppress_len])
        .repeat_dim(0, batch_size);
    let updates = token_ids.ones_like();
    let mask = Tensor::<B, 2, Int>::zeros([batch_size, vocab_size], device)
        .scatter(1, token_ids, updates, IndexingUpdateOp::Add)
        .bool();
    Some(mask)
}

pub fn apply_repetition_penalty<B: Backend>(
    logits: Tensor<B, 3>,
    past_token_ids: &Tensor<B, 2, Int>,
    penalty: Option<f32>,
) -> Tensor<B, 3> {
    let Some(penalty) = penalty else {
        return logits;
    };
    if !penalty.is_finite() || penalty <= 0.0 || (penalty - 1.0).abs() <= f32::EPSILON {
        return logits;
    }
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let history_len = past_token_ids.dims()[1];
    if history_len == 0 {
        return logits;
    }
    let logits_2d = logits.reshape([batch_size, vocab_size]);
    let gathered = logits_2d.clone().gather(1, past_token_ids.clone());
    let scale = 1.0 / penalty - 1.0;
    let deltas = gathered.mul_scalar(scale);
    let result_2d = logits_2d.scatter(1, past_token_ids.clone(), deltas, IndexingUpdateOp::Add);
    result_2d.reshape([batch_size, seq_len, vocab_size])
}
