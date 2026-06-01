use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub(crate) struct TimeDelayNetBlock<B: Backend> {
    pub(crate) conv: Conv1d<B>,
    #[module(skip)]
    pad_left: usize,
    #[module(skip)]
    pad_right: usize,
}

impl<B: Backend> TimeDelayNetBlock<B> {
    pub(crate) fn new(
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

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(
            self.conv
                .forward(reflect_pad_1d(x, self.pad_left, self.pad_right)),
        )
    }
}

fn reflect_pad_1d<B: Backend>(x: Tensor<B, 3>, pad_left: usize, pad_right: usize) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return x;
    }

    let [batch, channels, time] = x.dims();
    let mut segments =
        Vec::with_capacity(1 + usize::from(pad_left > 0) + usize::from(pad_right > 0));
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
