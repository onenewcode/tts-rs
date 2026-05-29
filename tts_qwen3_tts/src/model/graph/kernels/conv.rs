// Causal convolution primitives shared by talker and audio codec.
use burn::module::Module;
use burn::nn::conv::{Conv1d, ConvTranspose1d};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct AudioCodecCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct AudioCodecCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

impl<B: Backend> AudioCodecCausalConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}

impl<B: Backend> AudioCodecCausalTransConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.conv.forward(x);
        let trim = self.conv.kernel_size.saturating_sub(self.conv.stride);
        if trim == 0 {
            return output;
        }

        let [batch, channels, time] = output.dims();
        let end = time.saturating_sub(trim);
        output.slice([0..batch, 0..channels, trim..end])
    }
}

impl<B: Backend> AudioCodecCausalConv1d<B> {
    #[allow(clippy::too_many_arguments)]
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
        let pad_left = (kernel_size - 1) * dilation;
        use burn::nn::PaddingConfig1d;
        use burn::nn::conv::Conv1dConfig;
        Self {
            conv: Conv1dConfig::new(channels_in, channels_out, kernel_size)
                .with_stride(stride)
                .with_dilation(dilation)
                .with_groups(groups)
                .with_bias(bias)
                .with_padding(PaddingConfig1d::Explicit(pad_left, 0))
                .init(device),
        }
    }
}

impl<B: Backend> AudioCodecCausalTransConv1d<B> {
    pub(crate) fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        use burn::nn::conv::ConvTranspose1dConfig;
        Self {
            conv: ConvTranspose1dConfig::new([channels_in, channels_out], kernel_size)
                .with_stride(stride)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
        }
    }
}
