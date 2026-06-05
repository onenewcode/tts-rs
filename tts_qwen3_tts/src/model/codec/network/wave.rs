use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::Tensor;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;

use super::activation::AudioCodecSnakeBeta;
use super::conv::{AudioCodecCausalConv1d, AudioCodecCausalTransConv1d, conv1d_padding, pad_1d};

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecConvNeXtBlock<B: Backend> {
    pub dwconv: AudioCodecCausalConv1d<B>,
    pub norm: LayerNorm<B>,
    pub pwconv1: Linear<B>,
    pub pwconv2: Linear<B>,
    pub gamma: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3TtsAudioCodecWaveDecoderEntry<B: Backend> {
    InputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
    UpsampleStage(Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B>),
    OutputActivation(AudioCodecSnakeBeta<B>),
    OutputConv(Qwen3TtsAudioCodecWaveDecoderConvEntry<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecWaveDecoderConvEntry<B: Backend> {
    pub conv: Conv1d<B>,
    #[module(skip)]
    pub pad_mode: PadMode,
}

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

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderResidualUnit<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.act1.forward(hidden);
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        let hidden = self.conv2.forward(hidden);
        residual + hidden
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderUpsampleStage<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.block.0.forward(hidden);
        let hidden = self.block.1.forward(hidden);
        let hidden = self.block.2.forward(hidden);
        let hidden = self.block.3.forward(hidden);
        self.block.4.forward(hidden)
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderEntry<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::InputConv(entry) | Self::OutputConv(entry) => entry.forward(hidden),
            Self::UpsampleStage(stage) => stage.forward(hidden),
            Self::OutputActivation(snake) => snake.forward(hidden),
        }
    }
}

impl<B: Backend> Qwen3TtsAudioCodecWaveDecoderConvEntry<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let (padding_total, extra_padding) = conv1d_padding(&self.conv, hidden.dims()[2]);
        self.conv
            .forward(pad_1d(hidden, padding_total, extra_padding, self.pad_mode))
    }
}

impl<B: Backend> Qwen3TtsAudioCodecConvNeXtBlock<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let channels = hidden.dims()[1];
        let hidden = self.dwconv.forward(hidden).swap_dims(1, 2);
        let hidden = self.norm.forward(hidden);
        let hidden = self.pwconv1.forward(hidden);
        let hidden = gelu(hidden);
        let hidden = self.pwconv2.forward(hidden).swap_dims(1, 2);
        let gamma = self.gamma.val().reshape([1, channels, 1]);
        residual + hidden.mul(gamma)
    }
}
