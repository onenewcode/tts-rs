use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::DType;
use burn::tensor::Tensor;
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;

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
        let dtype = x.dtype();
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        self.down_proj
            .forward(silu(gate.cast(DType::F32)).cast(dtype) * up)
    }
}
// TODO 考虑使用burn的模块
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
        let x = native_linear_3d(&self.linear_fc1, x);
        let dtype = x.dtype();
        let x = silu(x.cast(DType::F32)).cast(dtype);
        native_linear_3d(&self.linear_fc2, x)
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
