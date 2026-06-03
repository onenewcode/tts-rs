mod pooling;
mod res2net;
mod tdnn;

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};

use self::pooling::AttentiveStatisticsPooling;
use self::res2net::SqueezeExcitationRes2NetBlock;
use self::tdnn::TimeDelayNetBlock;

use super::config::SpeakerEncoderConfig;

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
    pub(crate) fn new(config: &SpeakerEncoderConfig, device: &B::Device) -> Self {
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
        let mut hidden = Some(initial_tdnn.forward(mel));
        let se_blocks = &self.blocks[1..];
        let mut outputs = Vec::with_capacity(3);
        for (idx, block) in se_blocks.iter().enumerate() {
            let SpeakerEncoderBlock::Se(block) = block else {
                unreachable!("speaker encoder blocks 1..3 are SE-Res2Net blocks")
            };
            let current = block.forward(
                hidden
                    .take()
                    .expect("speaker encoder hidden state should be present"),
            );
            if idx + 1 == se_blocks.len() {
                outputs.push(current);
            } else {
                outputs.push(current.clone());
                hidden = Some(current);
            }
        }
        let hidden = self.mfa.forward(Tensor::cat(outputs, 1));
        let pooled = self.asp.forward(hidden);
        self.fc.forward(pooled).squeeze_dim(2)
    }

    pub(crate) fn dtype(&self) -> DType {
        let SpeakerEncoderBlock::Initial(initial_tdnn) = &self.blocks[0] else {
            unreachable!("speaker encoder block 0 is always the initial TDNN")
        };
        initial_tdnn.conv.weight.val().dequantize().dtype()
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum SpeakerEncoderBlock<B: Backend> {
    Initial(TimeDelayNetBlock<B>),
    Se(SqueezeExcitationRes2NetBlock<B>),
}
