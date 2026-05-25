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
    let x = x / (variance + norm.epsilon).sqrt();
    x.cast(dtype) * norm.gamma.val().cast(dtype).unsqueeze()
}
