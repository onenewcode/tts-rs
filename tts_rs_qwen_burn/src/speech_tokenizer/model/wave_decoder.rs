use burn::module::Module;
use burn::nn::conv::Conv1d;
use burn::tensor::backend::Backend;

use super::common::{TokenizerCausalConv1d, TokenizerCausalTransConv1d, TokenizerSnakeBeta};

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
