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

impl<B: Backend> Qwen3TtsTextMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = x.dtype();
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        self.down_proj.forward(silu(gate.cast(DType::F32)).cast(dtype) * up)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerResizeMlp<B: Backend> {
    pub linear_fc1: Linear<B>,
    pub linear_fc2: Linear<B>,
}

impl<B: Backend> Qwen3TtsTalkerResizeMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_fc1.forward(x);
        let dtype = x.dtype();
        let x = silu(x.cast(DType::F32)).cast(dtype);
        self.linear_fc2.forward(x)
    }
}
