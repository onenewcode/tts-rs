use burn::module::Module;
use burn::nn::conv::Conv1d;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::shared::nn::conv::{TokenizerCausalConv1d, TokenizerCausalTransConv1d};
use crate::shared::nn::activation::TokenizerSnakeBeta;

#[derive(Module, Debug)]
pub enum Qwen3TtsSpeechTokenizerWaveDecoderEntry<B: Backend> {
    InputConv(Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B>),
    UpsampleStage(Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage<B>),
    OutputActivation(TokenizerSnakeBeta<B>),
    OutputConv(Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage<B: Backend> {
    pub block: (
        TokenizerSnakeBeta<B>,
        TokenizerCausalTransConv1d<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
        Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B: Backend> {
    pub act1: TokenizerSnakeBeta<B>,
    pub conv1: TokenizerCausalConv1d<B>,
    pub act2: TokenizerSnakeBeta<B>,
    pub conv2: TokenizerCausalConv1d<B>,
}

// ---- Forward implementations ----

impl<B: Backend> Qwen3TtsSpeechTokenizerWaveDecoderConvEntry<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerWaveDecoderResidualUnit<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.act1.forward(x);
        let h = self.conv1.forward(h);
        let h = self.act2.forward(h);
        let h = self.conv2.forward(h);
        residual + h
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerWaveDecoderUpsampleStage<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.block.0.forward(x);
        let h = self.block.1.forward(h);
        let h = self.block.2.forward(h);
        let h = self.block.3.forward(h);
        let h = self.block.4.forward(h);
        h
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerWaveDecoderEntry<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::InputConv(entry) => entry.forward(x),
            Self::UpsampleStage(stage) => stage.forward(x),
            Self::OutputActivation(snake) => snake.forward(x),
            Self::OutputConv(entry) => entry.forward(x),
        }
    }
}
