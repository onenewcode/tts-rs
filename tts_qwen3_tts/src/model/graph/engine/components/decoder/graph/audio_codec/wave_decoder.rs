use burn::module::Module;
use burn::nn::conv::Conv1d;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::kernels::activation::AudioCodecSnakeBeta;
use crate::kernels::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d};

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3TtsAudioCodecWaveDecoderEntry<B: Backend> {
    InputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
    UpsampleStage(Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B>),
    OutputActivation(AudioCodecSnakeBeta<B>),
    OutputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
}
/// TODO 无意义的嵌套
#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecWaveDecoderConvEntry<B: Backend> {
    pub conv: Conv1d<B>,
}
/// TODO 该结构体设计有缺陷需要纠正
#[derive(Module, Debug)]
#[allow(clippy::type_complexity)]
pub struct Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B: Backend> {
    pub block: (
        AudioCodecSnakeBeta<B>,
        AudioCodecCausalTransConv1d<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
        Qwen3TtsAudioCodecWaveDecoderResidualUnit<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecWaveDecoderResidualUnit<B: Backend> {
    pub act1: AudioCodecSnakeBeta<B>,
    pub conv1: AudioCodecCausalConv1d<B>,
    pub act2: AudioCodecSnakeBeta<B>,
    pub conv2: AudioCodecCausalConv1d<B>,
}

// ---- Forward implementations ----

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderConvEntry<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}
/// TODO 经行优化
impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderResidualUnit<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.act1.forward(x);
        let h = self.conv1.forward(h);
        let h = self.act2.forward(h);
        let h = self.conv2.forward(h);
        residual + h
    }
}
/// TODO 经行优化
impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.block.0.forward(x);
        let h = self.block.1.forward(h);
        let h = self.block.2.forward(h);
        let h = self.block.3.forward(h);
        self.block.4.forward(h)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderEntry<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::InputConv(entry) => entry.forward(x),
            Self::UpsampleStage(stage) => stage.forward(x),
            Self::OutputActivation(snake) => snake.forward(x),
            Self::OutputConv(entry) => entry.forward(x),
        }
    }
}
