use burn::module::{Module, Param};
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

#[derive(Module, Debug)]
pub struct TokenizerSnakeBeta<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct TokenizerLayerScale<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug, Default, Clone)]
pub struct Qwen3TtsSpeechTokenizerEmptyModule;
