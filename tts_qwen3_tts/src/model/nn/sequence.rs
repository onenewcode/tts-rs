use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

pub(crate) fn select_last_sequence_step<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    let [_batch_size, seq_len, _hidden_size] = hidden.dims();
    if seq_len == 1 {
        return hidden;
    }
    let device = hidden.device();
    let last_step = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![i32::try_from(seq_len - 1).unwrap()], [1]),
        &device,
    );
    hidden.select(1, last_step)
}

#[cfg(test)]
mod tests {
    use super::select_last_sequence_step;
    use burn::backend::Flex;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn select_last_sequence_step_returns_last_time_slice() {
        let device = Default::default();
        let hidden = Tensor::<Flex, 3>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 3, 2]),
            &device,
        );

        let last = select_last_sequence_step(hidden);
        let values = last
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("selected step should be readable");

        assert_eq!(values, vec![5.0, 6.0]);
    }

    #[test]
    fn select_last_sequence_step_preserves_single_step_tensor() {
        let device = Default::default();
        let hidden =
            Tensor::<Flex, 3>::from_data(TensorData::new(vec![7.0_f32, 8.0], [1, 1, 2]), &device);

        let last = select_last_sequence_step(hidden);
        let values = last
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("selected step should be readable");

        assert_eq!(values, vec![7.0, 8.0]);
    }
}
