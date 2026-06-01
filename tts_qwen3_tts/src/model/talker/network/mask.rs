use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::model::nn::attention::autoregressive_attention_mask;
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
    let causal_mask = (query_len == key_len)
        .then(|| autoregressive_attention_mask::<B>(batch_size, query_len, device));

    let padding_mask =
        padding_mask.map(|mask| mask.equal_elem(0).unsqueeze::<4>().repeat_dim(2, query_len));

    match (causal_mask, padding_mask) {
        (Some(causal), Some(padding)) => Some(causal.bool_or(padding)),
        (Some(causal), None) => Some(causal),
        (None, Some(padding)) => Some(padding),
        (None, None) => None,
    }
}
