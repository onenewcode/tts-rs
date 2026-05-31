use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;

pub(crate) fn flatten_batch_sequence<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, feature_size] = tensor.dims();
    tensor.reshape([batch_size * seq_len, feature_size])
}

pub(crate) fn unflatten_batch_sequence<B: Backend>(
    tensor: Tensor<B, 2>,
    batch_size: usize,
    seq_len: usize,
) -> Tensor<B, 3> {
    let [_flat, feature_size] = tensor.dims();
    tensor.reshape([batch_size, seq_len, feature_size])
}

pub(crate) fn read_float_tensor_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    context: &str,
) -> Result<Vec<f32>, QwenTtsInferenceError> {
    tensor
        .try_into_data()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("{context}: {source}"),
        })?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("{context}: {source}"),
        })
}

pub(crate) fn read_int_tensor_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
    context: &str,
) -> Result<Vec<i64>, QwenTtsInferenceError> {
    tensor
        .try_into_data()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("{context}: {source}"),
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("{context}: {source}"),
        })
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_batch_sequence, read_float_tensor_vec, read_int_tensor_vec,
        unflatten_batch_sequence,
    };
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor, TensorData};

    #[test]
    fn flatten_batch_sequence_round_trips_through_unflatten() {
        let device = Default::default();
        let hidden = Tensor::<Flex, 3>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );

        let flat = flatten_batch_sequence(hidden);
        assert_eq!(flat.dims(), [2, 2]);

        let restored = unflatten_batch_sequence(flat, 1, 2);
        assert_eq!(restored.dims(), [1, 2, 2]);
    }

    #[test]
    fn read_tensor_vec_helpers_convert_float_and_int_tensors() {
        let device = Default::default();
        let float_tensor = Tensor::<Flex, 1>::from_data(TensorData::from([1.0_f32, 2.0]), &device);
        let int_tensor =
            Tensor::<Flex, 2, Int>::from_data(TensorData::new(vec![3_i64, 4_i64], [1, 2]), &device);

        let floats =
            read_float_tensor_vec(float_tensor, "float test").expect("float tensor should read");
        let ints = read_int_tensor_vec(int_tensor, "int test").expect("int tensor should read");

        assert_eq!(floats, vec![1.0, 2.0]);
        assert_eq!(ints, vec![3, 4]);
    }
}
