use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};

use super::config::SpeakerEncoderConfigManifest;

#[derive(Module, Debug)]
pub(crate) struct SpeakerEncoderNetwork<B: Backend> {
    blocks: Vec<SpeakerEncoderBlock<B>>,
    mfa: TimeDelayNetBlock<B>,
    asp: AttentiveStatisticsPooling<B>,
    fc: Conv1d<B>,
    #[module(skip)]
    pub(crate) enc_dim: usize,
}

impl<B: Backend> SpeakerEncoderNetwork<B> {
    pub(crate) fn new(config: &SpeakerEncoderConfigManifest, device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(4);
        blocks.push(SpeakerEncoderBlock::Initial(TimeDelayNetBlock::new(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            device,
        )));
        for idx in 1..4 {
            blocks.push(SpeakerEncoderBlock::Se(SqueezeExcitationRes2NetBlock::new(
                config.enc_channels[idx],
                config.enc_kernel_sizes[idx],
                config.enc_dilations[idx],
                config.enc_res2net_scale,
                config.enc_se_channels,
                device,
            )));
        }

        let mfa_in_channels: usize = config.enc_channels[1..4].iter().sum();
        Self {
            blocks,
            mfa: TimeDelayNetBlock::new(
                mfa_in_channels,
                config.enc_channels[4],
                config.enc_kernel_sizes[4],
                config.enc_dilations[4],
                device,
            ),
            asp: AttentiveStatisticsPooling::new(
                config.enc_channels[4],
                config.enc_attention_channels,
                device,
            ),
            fc: Conv1dConfig::new(config.enc_channels[4] * 2, config.enc_dim, 1)
                .with_bias(true)
                .init(device),
            enc_dim: config.enc_dim,
        }
    }

    pub(crate) fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 2> {
        let SpeakerEncoderBlock::Initial(initial_tdnn) = &self.blocks[0] else {
            unreachable!("speaker encoder block 0 is always the initial TDNN")
        };
        let mut hidden = initial_tdnn.forward(mel);
        let mut outputs = Vec::with_capacity(3);
        for block in &self.blocks[1..] {
            let SpeakerEncoderBlock::Se(block) = block else {
                unreachable!("speaker encoder blocks 1..3 are SE-Res2Net blocks")
            };
            hidden = block.forward(hidden);
            outputs.push(hidden.clone());
        }
        let hidden = self.mfa.forward(Tensor::cat(outputs, 1));
        let pooled = self.asp.forward(hidden);
        self.fc.forward(pooled).squeeze_dim(2)
    }

    pub(crate) fn dtype(&self) -> DType {
        let SpeakerEncoderBlock::Initial(initial_tdnn) = &self.blocks[0] else {
            unreachable!("speaker encoder block 0 is always the initial TDNN")
        };
        initial_tdnn.conv.weight.val().dtype()
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum SpeakerEncoderBlock<B: Backend> {
    Initial(TimeDelayNetBlock<B>),
    Se(SqueezeExcitationRes2NetBlock<B>),
}

#[derive(Module, Debug)]
struct TimeDelayNetBlock<B: Backend> {
    conv: Conv1d<B>,
    #[module(skip)]
    pad_left: usize,
    #[module(skip)]
    pad_right: usize,
}

impl<B: Backend> TimeDelayNetBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let total_pad = dilation * (kernel_size - 1);
        Self {
            conv: Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true)
                .init(device),
            pad_left: total_pad / 2,
            pad_right: total_pad - total_pad / 2,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(
            self.conv
                .forward(reflect_pad_1d(x, self.pad_left, self.pad_right)),
        )
    }
}

#[derive(Module, Debug)]
struct Res2NetBlock<B: Backend> {
    blocks: Vec<TimeDelayNetBlock<B>>,
    #[module(skip)]
    scale: usize,
    #[module(skip)]
    chunk_size: usize,
}

impl<B: Backend> Res2NetBlock<B> {
    fn new(
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

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut outputs = Vec::with_capacity(self.scale);
        outputs.push(
            x.clone()
                .slice([0..x.dims()[0], 0..self.chunk_size, 0..x.dims()[2]]),
        );
        for (idx, block) in self.blocks.iter().enumerate() {
            let chunk = x.clone().slice([
                0..x.dims()[0],
                (idx + 1) * self.chunk_size..(idx + 2) * self.chunk_size,
                0..x.dims()[2],
            ]);
            let input = if idx == 0 {
                chunk
            } else {
                chunk + outputs.last().expect("previous Res2Net chunk").clone()
            };
            outputs.push(block.forward(input));
        }
        Tensor::cat(outputs, 1)
    }
}

#[derive(Module, Debug)]
struct SqueezeExcitationBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> SqueezeExcitationBlock<B> {
    fn new(channels: usize, se_channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(channels, se_channels, 1)
                .with_bias(true)
                .init(device),
            conv2: Conv1dConfig::new(se_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = x.clone().mean_dim(2);
        let scale = relu(self.conv1.forward(scale));
        let scale = sigmoid(self.conv2.forward(scale));
        x * scale
    }
}

#[derive(Module, Debug)]
struct SqueezeExcitationRes2NetBlock<B: Backend> {
    tdnn1: TimeDelayNetBlock<B>,
    res2net_block: Res2NetBlock<B>,
    tdnn2: TimeDelayNetBlock<B>,
    se_block: SqueezeExcitationBlock<B>,
}

impl<B: Backend> SqueezeExcitationRes2NetBlock<B> {
    fn new(
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

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let hidden = self.tdnn1.forward(x);
        let hidden = self.res2net_block.forward(hidden);
        let hidden = self.tdnn2.forward(hidden);
        let hidden = self.se_block.forward(hidden);
        hidden + residual
    }
}

#[derive(Module, Debug)]
struct AttentiveStatisticsPooling<B: Backend> {
    tdnn: TimeDelayNetBlock<B>,
    conv: Conv1d<B>,
}

impl<B: Backend> AttentiveStatisticsPooling<B> {
    fn new(channels: usize, attention_channels: usize, device: &B::Device) -> Self {
        Self {
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, device),
            conv: Conv1dConfig::new(attention_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let mean = x.clone().mean_dim(2);
        let diff = x.clone() - mean.clone();
        let var = diff.powi_scalar(2).mean_dim(2);
        let std = (var + 1e-5).sqrt();
        let attn_in = Tensor::cat(
            vec![
                x.clone(),
                mean.clone().expand([batch, channels, time]),
                std.expand([batch, channels, time]),
            ],
            1,
        );
        let attn = self.tdnn.forward(attn_in).tanh();
        let attn = softmax(self.conv.forward(attn), 2);
        let weighted_mean = (x.clone() * attn.clone()).sum_dim(2);
        let weighted_diff = x - weighted_mean.clone();
        let weighted_var = (weighted_diff.powi_scalar(2) * attn).sum_dim(2);
        let weighted_std = (weighted_var + 1e-5).sqrt();
        Tensor::cat(vec![weighted_mean, weighted_std], 1)
    }
}

fn reflect_pad_1d<B: Backend>(x: Tensor<B, 3>, pad_left: usize, pad_right: usize) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return x;
    }

    let [batch, channels, time] = x.dims();
    let mut segments = Vec::with_capacity(3);
    if pad_left > 0 {
        segments.push(
            x.clone()
                .slice([0..batch, 0..channels, 1..pad_left + 1])
                .flip([2]),
        );
    }
    segments.push(x.clone());
    if pad_right > 0 {
        segments.push(
            x.slice([0..batch, 0..channels, time - 1 - pad_right..time - 1])
                .flip([2]),
        );
    }
    Tensor::cat(segments, 2)
}
