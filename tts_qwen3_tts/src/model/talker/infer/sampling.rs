//! Configurable token sampling strategies for autoregressive generation.
//!
//! Supports greedy argmax and randomized sampling with:
//! suppress -> temperature -> top-k -> top-p -> softmax -> categorical.

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor};

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
    let logits_2d = prepare_last_step_logits(logits);
    if suppress_token_ids.is_empty() {
        return sample_token_from_logits(logits_2d, sampling, None);
    }

    let [batch_size, vocab_size] = logits_2d.dims();
    let device = logits_2d.device();
    let suppress_mask =
        suppress_token_mask::<B>(batch_size, vocab_size, suppress_token_ids, &device);

    sample_token_from_logits(logits_2d, sampling, suppress_mask.as_ref())
}

fn sample_token_from_logits<B: Backend>(
    mut logits_2d: Tensor<B, 2>,
    sampling: &SamplingConfig,
    suppress_mask: Option<&Tensor<B, 2, Bool>>,
) -> Tensor<B, 2, Int> {
    if let Some(suppress_mask) = suppress_mask {
        logits_2d = logits_2d.mask_fill(suppress_mask.clone(), f32::NEG_INFINITY);
    }

    if !sampling.do_sample {
        return logits_2d.argmax(1);
    }

    let device = logits_2d.device();
    let stable_dtype = Tensor::<B, 1>::zeros([1], &device).dtype();
    logits_2d = logits_2d.dequantize().cast(stable_dtype);
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

    softmax(logits_2d, 1).categorical(1)
}

fn prepare_last_step_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    if seq_len == 1 {
        logits.reshape([batch_size, vocab_size])
    } else {
        select_last_sequence_step(logits).reshape([batch_size, vocab_size])
    }
}

pub(super) fn suppress_token_mask<B: Backend>(
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

pub(crate) fn repetition_penalty_enabled(penalty: Option<f32>) -> bool {
    penalty.is_some_and(|penalty| {
        penalty.is_finite() && penalty > 0.0 && (penalty - 1.0).abs() > f32::EPSILON
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::{DType, Tensor};

    use super::{SamplingConfig, repetition_penalty_enabled, sample_token};
    use crate::loading::runtime::RuntimeBackend;

    #[test]
    fn sample_token_skips_masked_logits() {
        let device = Default::default();
        let logits =
            Tensor::<RuntimeBackend, 1>::from_floats([0.0, 1.0, 2.0, 9.0].as_slice(), &device)
                .reshape([1, 1, 4]);

        let selected = sample_token(logits, &SamplingConfig::default(), &[3]);
        let values = selected
            .try_into_data()
            .expect("selected token should be readable")
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("selected token should convert to vec");

        assert_eq!(values, vec![2]);
    }

    #[test]
    fn repetition_penalty_enabled_only_accepts_meaningful_positive_values() {
        assert!(!repetition_penalty_enabled(None));
        assert!(!repetition_penalty_enabled(Some(1.0)));
        assert!(!repetition_penalty_enabled(Some(0.0)));
        assert!(!repetition_penalty_enabled(Some(f32::NAN)));
        assert!(repetition_penalty_enabled(Some(1.2)));
    }

    #[test]
    fn sampling_path_accepts_all_runtime_float_dtypes() {
        let device = Default::default();
        let sampling = SamplingConfig {
            do_sample: true,
            temperature: 0.7,
            top_k: Some(1),
            top_p: 0.8,
            repetition_penalty: None,
        };

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let logits =
                Tensor::<RuntimeBackend, 1>::from_floats([0.0, 1.0, 2.0, 9.0].as_slice(), &device)
                    .reshape([1, 1, 4])
                    .cast(dtype);

            let selected = sample_token(logits, &sampling, &[]);
            let values = selected
                .try_into_data()
                .expect("selected token should be readable")
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("selected token should convert to vec");

            assert_eq!(values, vec![3]);
        }
    }
}
