use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;

use super::tdnn::TimeDelayNetBlock;

#[derive(Module, Debug)]
pub(crate) struct Res2NetBlock<B: Backend> {
    blocks: Vec<TimeDelayNetBlock<B>>,
    #[module(skip)]
    scale: usize,
    #[module(skip)]
    chunk_size: usize,
}

impl<B: Backend> Res2NetBlock<B> {
    pub(crate) fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_size = channels / scale;
        let blocks = (0..scale - 1)
            .map(|_| TimeDelayNetBlock::new(chunk_size, chunk_size, kernel_size, dilation, device))
            .collect();
        Self {
            blocks,
            scale,
            chunk_size,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let channel_count = x.dims()[1];
        debug_assert_eq!(channel_count, self.chunk_size * self.scale);
        let mut outputs = Vec::with_capacity(self.scale);
        let mut chunks = x.chunk(self.scale, 1).into_iter();
        let mut previous = chunks
            .next()
            .expect("res2net input should include the first chunk");

        for (idx, (chunk, block)) in chunks.zip(self.blocks.iter()).enumerate() {
            let input = if idx == 0 {
                chunk
            } else {
                chunk + previous.clone()
            };
            let output = block.forward(input);
            outputs.push(previous);
            previous = output;
        }
        outputs.push(previous);
        Tensor::cat(outputs, 1)
    }
}

#[derive(Module, Debug)]
pub(crate) struct SqueezeExcitationBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> SqueezeExcitationBlock<B> {
    pub(crate) fn new(channels: usize, se_channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(channels, se_channels, 1)
                .with_bias(true)
                .init(device),
            conv2: Conv1dConfig::new(se_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = x.clone().mean_dim(2);
        let scale = relu(self.conv1.forward(scale));
        let scale = sigmoid(self.conv2.forward(scale));
        x * scale
    }
}

#[derive(Module, Debug)]
pub(crate) struct SqueezeExcitationRes2NetBlock<B: Backend> {
    tdnn1: TimeDelayNetBlock<B>,
    res2net_block: Res2NetBlock<B>,
    tdnn2: TimeDelayNetBlock<B>,
    se_block: SqueezeExcitationBlock<B>,
}

impl<B: Backend> SqueezeExcitationRes2NetBlock<B> {
    pub(crate) fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        se_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            tdnn1: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            res2net_block: Res2NetBlock::new(channels, kernel_size, dilation, scale, device),
            tdnn2: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            se_block: SqueezeExcitationBlock::new(channels, se_channels, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let hidden = self.tdnn1.forward(x);
        let hidden = self.res2net_block.forward(hidden);
        let hidden = self.tdnn2.forward(hidden);
        let hidden = self.se_block.forward(hidden);
        hidden + residual
    }
}
