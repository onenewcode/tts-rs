// Causal convolution primitives shared by talker and speech tokenizer.
use burn::module::Module;
use burn::nn::conv::{Conv1d, ConvTranspose1d};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct TokenizerCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct TokenizerCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

impl<B: Backend> TokenizerCausalConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}

impl<B: Backend> TokenizerCausalTransConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}
