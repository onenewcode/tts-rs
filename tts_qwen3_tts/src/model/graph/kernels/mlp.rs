use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::DType;
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

use crate::profiling::record_operator;

#[derive(Module, Debug)]
pub struct Qwen3TtsTextMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B> Qwen3TtsTextMlp<B>
where
    B: Backend,
{
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = record_operator("mlp.gate_proj", || self.gate_proj.forward(x.clone()));
        let up = record_operator("mlp.up_proj", || self.up_proj.forward(x));
        let activated = record_operator("mlp.activation", || activation::silu(gate));
        record_operator("mlp.down_proj", || self.down_proj.forward(activated * up))
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerResizeMlp<B: Backend> {
    pub linear_fc1: Linear<B>,
    pub linear_fc2: Linear<B>,
}

impl<B> Qwen3TtsTalkerResizeMlp<B>
where
    B: Backend,
{
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = record_operator("mlp.resize_fc1", || native_linear_3d(&self.linear_fc1, x));
        let x = record_operator("mlp.resize_activation", || activation::silu(x));
        record_operator("mlp.resize_fc2", || native_linear_3d(&self.linear_fc2, x))
    }
}

pub(crate) fn native_linear_3d<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, in_features] = x.dims();
    let out_features = linear.weight.dims()[1];
    let x_2d = x.reshape([batch_size * seq_len, in_features]);

    match &linear.bias {
        Some(bias) => {
            let dtype = x_2d.dtype();
            let ones = Tensor::<B, 2>::ones([batch_size * seq_len, 1], &x_2d.device()).cast(dtype);
            let x_aug = Tensor::cat(vec![x_2d, ones], 1);
            let w_aug = Tensor::cat(vec![linear.weight.val(), bias.val().unsqueeze::<2>()], 0);
            x_aug
                .matmul(w_aug)
                .reshape([batch_size, seq_len, out_features])
        }
        None => linear
            .forward(x_2d)
            .reshape([batch_size, seq_len, out_features]),
    }
}
