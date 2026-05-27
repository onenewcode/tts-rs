//! Configurable token sampling strategies for autoregressive generation.
//!
//! Supports greedy argmax and randomized sampling with:
//! suppress → temperature → top-k → top-p → softmax → categorical.
//!
//! All operations stay on-device using Burn tensor APIs.

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData};

/// Controls how tokens are selected from logits during generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: None,
            repetition_penalty: None,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            ..Default::default()
        }
    }
}

/// Select one token per batch item from the last position of logits.
pub fn sample_token<B: Backend>(
    logits: Tensor<B, 3>,
    sampling: &SamplingConfig,
    eos_token_id: Option<usize>,
    suppress_token_ids: &[usize],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Bool>) {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let mut logits_2d = logits
        .slice([0..batch_size, seq_len - 1..seq_len, 0..vocab_size])
        .reshape([batch_size, vocab_size]);

    // 1. Suppress tokens
    if !suppress_token_ids.is_empty() {
        let mut mask_data = vec![false; batch_size * vocab_size];
        for batch in 0..batch_size {
            for &id in suppress_token_ids {
                if id < vocab_size {
                    mask_data[batch * vocab_size + id] = true;
                }
            }
        }
        let suppress_mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, [batch_size, vocab_size]),
            device,
        );
        logits_2d = logits_2d.mask_fill(suppress_mask, f32::NEG_INFINITY);
    }

    if !sampling.do_sample {
        let selected = greedy_argmax_lowest_index(logits_2d, batch_size, vocab_size, device);
        let eos_mask = match eos_token_id {
            Some(id) => selected.clone().equal_elem(id as i64).reshape([batch_size]),
            None => Tensor::<B, 1, Bool>::zeros([batch_size], device),
        };
        return (selected, eos_mask);
    }

    // 2. Temperature
    logits_2d = logits_2d.div_scalar(sampling.temperature.max(1e-5));

    // 3. Top-k
    if let Some(k) = sampling.top_k.filter(|k| *k > 0 && *k < vocab_size) {
        let kth_value = logits_2d
            .clone()
            .topk(k, 1)
            .slice([0..batch_size, k - 1..k]);
        let mask = logits_2d.clone().lower(kth_value);
        logits_2d = logits_2d.mask_fill(mask, f32::NEG_INFINITY);
    }

    // 4. Top-p
    if sampling.top_p < 1.0 {
        let (sorted_vals, sorted_idx) = logits_2d.clone().sort_descending_with_indices(1);
        let sorted_probs = softmax(sorted_vals.clone().cast(DType::F32), 1).cast(DType::F32);
        let cumsum = sorted_probs.clone().cumsum(1);
        let sorted_keep: Tensor<B, 2, Bool> = cumsum.sub(sorted_probs).lower_elem(sampling.top_p);
        let inverse = sorted_idx.argsort(1);
        let orig_keep: Tensor<B, 2, Bool> = sorted_keep.gather(1, inverse);
        logits_2d = logits_2d.mask_fill(orig_keep.bool_not(), f32::NEG_INFINITY);
    }

    // 5. Softmax + categorical
    let probs = softmax(logits_2d.clone().cast(DType::F32), 1);
    let selected = probs.categorical(1);

    let eos_mask = match eos_token_id {
        Some(id) => selected.clone().equal_elem(id as i64).reshape([batch_size]),
        None => Tensor::<B, 1, Bool>::zeros([batch_size], device),
    };

    (selected, eos_mask)
}

fn greedy_argmax_lowest_index<B: Backend>(
    logits: Tensor<B, 2>,
    batch_size: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values = logits
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("greedy logits should be convertible to f32");
    let mut selected = Vec::with_capacity(batch_size);
    for row in values.chunks(vocab_size) {
        let mut best_id = 0_i32;
        let mut best_value = f32::NEG_INFINITY;
        for (id, value) in row.iter().copied().enumerate() {
            if value > best_value {
                best_id = id as i32;
                best_value = value;
            }
        }
        selected.push(best_id);
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(selected, [batch_size, 1]), device)
}

/// Apply repetition penalty: `logits[:, t] /= penalty` for each past token `t`.
/// Uses gather + scatter-add to stay on-device.
pub fn apply_repetition_penalty<B: Backend>(
    logits: Tensor<B, 3>,
    past_token_ids: &Tensor<B, 2, Int>,
    penalty: Option<f32>,
) -> Tensor<B, 3> {
    let Some(penalty) = penalty else {
        return logits;
    };
    if penalty == 1.0 {
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
