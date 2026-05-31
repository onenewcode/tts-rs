use burn::tensor::backend::Backend;
use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor, TensorData};

pub(crate) fn suppress_token_mask<B: Backend>(
    batch_size: usize,
    vocab_size: usize,
    suppress_token_ids: &[usize],
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    let valid_ids = suppress_token_ids
        .iter()
        .copied()
        .filter(|id| *id < vocab_size)
        .map(|id| id as i64)
        .collect::<Vec<_>>();
    if valid_ids.is_empty() {
        return None;
    }

    let suppress_len = valid_ids.len();
    let token_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(valid_ids, [1, suppress_len]), device)
            .repeat_dim(0, batch_size);
    let updates = token_ids.ones_like().float();
    let mask = Tensor::<B, 2>::zeros([batch_size, vocab_size], device)
        .scatter(1, token_ids, updates, IndexingUpdateOp::Add)
        .bool();
    Some(mask)
}

#[cfg(test)]
mod tests {
    use super::suppress_token_mask;
    use burn::backend::Flex;

    #[test]
    fn suppress_token_mask_marks_requested_ids() {
        let device = Default::default();
        let mask = suppress_token_mask::<Flex>(1, 5, &[1, 3, 9], &device)
            .expect("in-range ids should create a mask");
        let values = mask
            .into_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .expect("mask should be readable");

        assert_eq!(values, vec![false, true, false, true, false]);
    }

    #[test]
    fn suppress_token_mask_ignores_out_of_range_ids() {
        let device = Default::default();
        let mask = suppress_token_mask::<Flex>(2, 3, &[7], &device);

        assert!(mask.is_none());
    }
}
