use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Tensor};

pub(crate) fn apply_attention_kernel<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<&Tensor<B, 4, Bool>>,
    scale: f32,
    output_dtype: DType,
) -> Tensor<B, 4> {
    let stable_dtype = Tensor::<B, 1>::zeros([1], &query.device()).dtype();
    let query = query.dequantize().cast(stable_dtype);
    let key = key.dequantize().cast(stable_dtype);
    let value = value.dequantize().cast(stable_dtype);

    let scores = query.matmul(key.swap_dims(2, 3)).mul_scalar(scale);
    let scores = if let Some(mask) = mask {
        scores.mask_fill(mask.clone(), f32::NEG_INFINITY)
    } else {
        scores
    };
    let output = softmax(scores, 3).matmul(value);

    output.cast(output_dtype)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{DType, Int, Tensor};

    use crate::loading::runtime::RuntimeBackend;

    use super::apply_attention_kernel;

    #[test]
    fn attention_restores_requested_dtype() {
        let device = Default::default();

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let query = Tensor::<RuntimeBackend, 1>::from_floats(
                [1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5],
                &device,
            )
            .reshape([1, 2, 2, 2])
            .cast(dtype);
            let key = Tensor::<RuntimeBackend, 1>::from_floats(
                [1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -0.5, 0.5],
                &device,
            )
            .reshape([1, 2, 2, 2])
            .cast(dtype);
            let value = Tensor::<RuntimeBackend, 1>::from_floats(
                [0.5, 1.5, -1.0, 0.0, 1.0, -1.0, 0.25, 0.75],
                &device,
            )
            .reshape([1, 2, 2, 2])
            .cast(dtype);
            let mask = Tensor::<RuntimeBackend, 2, Int>::from_ints([[0, 1], [0, 0]], &device)
                .reshape([1, 1, 2, 2])
                .bool();

            let output = apply_attention_kernel(
                query,
                key,
                value,
                Some(&mask),
                (2.0f32).sqrt().recip(),
                dtype,
            );

            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), [1, 2, 2, 2]);
            let _ = output
                .try_into_data()
                .expect("attention output should be readable");
        }
    }
}
