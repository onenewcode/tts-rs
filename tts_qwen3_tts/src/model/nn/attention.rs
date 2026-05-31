use burn::nn::attention::generate_autoregressive_mask;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

pub(crate) fn repeat_kv_heads<B: Backend>(
    hidden: Tensor<B, 4>,
    repetitions: usize,
) -> Tensor<B, 4> {
    if repetitions == 1 {
        return hidden;
    }
    let [batch_size, num_kv_heads, seq_len, head_dim] = hidden.dims();
    hidden
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repetitions)
        .reshape([batch_size, num_kv_heads * repetitions, seq_len, head_dim])
}

pub(crate) fn autoregressive_attention_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    generate_autoregressive_mask::<B>(batch_size, seq_len, device).unsqueeze_dim::<4>(1)
}

#[cfg(test)]
mod tests {
    use super::{autoregressive_attention_mask, repeat_kv_heads};
    use burn::backend::Flex;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn repeat_kv_heads_repeats_each_head_group() {
        let device = Default::default();
        let hidden = Tensor::<Flex, 4>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 1, 2]),
            &device,
        );

        let repeated = repeat_kv_heads(hidden, 2);
        let values = repeated
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("repeated heads should be readable");

        assert_eq!(values, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_kv_heads_keeps_tensor_when_rep_is_one() {
        let device = Default::default();
        let hidden = Tensor::<Flex, 4>::from_data(
            TensorData::new(vec![1.0_f32, 2.0], [1, 1, 1, 2]),
            &device,
        );

        let repeated = repeat_kv_heads(hidden, 1);
        let values = repeated
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor should be readable");

        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn autoregressive_attention_mask_expands_to_four_dimensions() {
        let device = Default::default();
        let mask = autoregressive_attention_mask::<Flex>(1, 3, &device);
        let values = mask
            .into_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .expect("mask should be readable");

        assert_eq!(
            values,
            vec![
                false, true, true, //
                false, false, true, //
                false, false, false,
            ]
        );
    }
}
