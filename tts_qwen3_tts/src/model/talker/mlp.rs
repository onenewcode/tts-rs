use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        let activated = activation::silu(gate);
        self.down_proj.forward(activated * up)
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
        let x = self.linear_fc1.forward(x);
        let x = activation::silu(x);
        self.linear_fc2.forward(x)
    }
}
