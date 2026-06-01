use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;

use super::tdnn::TimeDelayNetBlock;

#[derive(Module, Debug)]
pub(crate) struct AttentiveStatisticsPooling<B: Backend> {
    tdnn: TimeDelayNetBlock<B>,
    conv: Conv1d<B>,
}

impl<B: Backend> AttentiveStatisticsPooling<B> {
    pub(crate) fn new(channels: usize, attention_channels: usize, device: &B::Device) -> Self {
        Self {
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, device),
            conv: Conv1dConfig::new(attention_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }
    // TODO 有没有更加高效的计算方式
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let mean = x.clone().mean_dim(2);
        let std = ((x.clone() - mean.clone()).powi_scalar(2).mean_dim(2) + 1e-5).sqrt();
        let attn_in = Tensor::cat(
            vec![
                x.clone(),
                mean.expand([batch, channels, time]),
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
