use burn::nn::RmsNorm;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};

pub(crate) fn qwen_rms_norm<B: Backend, const D: usize>(
    norm: &RmsNorm<B>,
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    let dtype = x.dtype();
    let x = x.cast(DType::F32);
    let variance = x.clone().square().mean_dim(D - 1);
    let x = x * (variance + norm.epsilon).sqrt().recip();
    norm.gamma.val().cast(dtype).unsqueeze() * cast_with_bf16_tie_bias(x, dtype)
}

fn cast_with_bf16_tie_bias<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dtype: DType,
) -> Tensor<B, D> {
    if dtype != DType::BF16 {
        return x.cast(dtype);
    }

    let positive = x.clone() + 1.0e-6;
    let negative = x.clone() - 1.0e-6;
    positive
        .mask_where(x.lower_elem(0.0), negative)
        .cast(DType::BF16)
}
