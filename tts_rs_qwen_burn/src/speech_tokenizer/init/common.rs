use burn::module::Initializer;
use burn::nn::conv::{Conv1dConfig, ConvTranspose1dConfig};
use burn::tensor::backend::Backend;

#[cfg(test)]
use burn::module::Param;
#[cfg(test)]
use burn::tensor::Tensor;

use crate::speech_tokenizer::{
    Qwen3TtsSpeechTokenizerDecoderCodebook, Qwen3TtsSpeechTokenizerEncoderCodebook,
    TokenizerCausalConv1d, TokenizerCausalTransConv1d, TokenizerLayerScale, TokenizerSnakeBeta,
};

impl<B: Backend> Qwen3TtsSpeechTokenizerDecoderCodebook<B> {
    pub(crate) fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embedding_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}

impl<B: Backend> Qwen3TtsSpeechTokenizerEncoderCodebook<B> {
    pub(crate) fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            initialized: Initializer::Ones.init([1], device),
            cluster_usage: Initializer::Ones.init([codebook_size], device),
            embed_sum: Initializer::Zeros.init([codebook_size, dim], device),
        }
    }
}

impl<B: Backend> TokenizerCausalConv1d<B> {
    pub(crate) fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: Conv1dConfig::new(channels_in, channels_out, kernel_size)
                .with_stride(stride)
                .with_dilation(dilation)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
        }
    }
}

impl<B: Backend> TokenizerCausalTransConv1d<B> {
    pub(crate) fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: ConvTranspose1dConfig::new([channels_in, channels_out], kernel_size)
                .with_stride(stride)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
        }
    }
}

impl<B: Backend> TokenizerSnakeBeta<B> {
    pub(crate) fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            alpha: Initializer::Zeros.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }
}

impl<B: Backend> TokenizerLayerScale<B> {
    pub(crate) fn new(channels: usize, initial_scale: f64, device: &B::Device) -> Self {
        Self {
            scale: Initializer::Constant {
                value: initial_scale,
            }
            .init([channels], device),
        }
    }
}

#[cfg(test)]
pub(crate) fn tensor_param_dims<const D: usize, B: Backend>(
    param: &Param<Tensor<B, D>>,
) -> [usize; D] {
    param.dims()
}
