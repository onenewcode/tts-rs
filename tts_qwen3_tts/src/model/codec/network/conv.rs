// Audio codec causal convolution wrappers around Burn conv modules.
use burn::module::Module;
use burn::nn::conv::{Conv1d, ConvTranspose1d};
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug, Clone, Copy)]
pub(crate) enum ConvPadMode {
    Constant,
    Replicate,
}

#[derive(Module, Debug)]
pub struct AudioCodecCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
    #[module(skip)]
    pad_mode: ConvPadMode,
}

#[derive(Module, Debug)]
pub struct AudioCodecCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

impl<B: Backend> AudioCodecCausalConv1d<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        forward_padded_conv1d(&self.conv, self.pad_mode, hidden)
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

pub(crate) fn forward_padded_conv1d<B: Backend>(
    conv: &Conv1d<B>,
    pad_mode: ConvPadMode,
    hidden: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let time_steps = hidden.dims()[2];
    let effective_kernel = (conv.kernel_size - 1) * conv.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(conv.stride);
    let extra_padding =
        extra_padding_for_conv1d(time_steps, effective_kernel, conv.stride, padding_total);
    let hidden = pad_1d(hidden, padding_total, extra_padding, pad_mode);
    conv.forward(hidden)
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
        pad_mode: ConvPadMode,
        device: &B::Device,
    ) -> Self {
        use burn::nn::conv::Conv1dConfig;
        Self {
            conv: Conv1dConfig::new(channels_in, channels_out, kernel_size)
                .with_stride(stride)
                .with_dilation(dilation)
                .with_groups(groups)
                .with_bias(bias)
                .init(device),
            pad_mode,
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

fn extra_padding_for_conv1d(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let padded_len = len.saturating_add(padding_total);
    let missing = kernel_size.saturating_sub(padded_len);
    if missing > 0 {
        missing
    } else {
        (stride - (padded_len - kernel_size) % stride) % stride
    }
}

fn pad_1d<B: Backend>(
    hidden: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
    mode: ConvPadMode,
) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return hidden;
    }
    match mode {
        ConvPadMode::Constant => hidden.pad((pad_left, pad_right, 0, 0), PadMode::Constant(0.0)),
        ConvPadMode::Replicate => replicate_pad_1d(hidden, pad_left, pad_right),
    }
}

fn replicate_pad_1d<B: Backend>(
    hidden: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
) -> Tensor<B, 3> {
    let [batch, channels, time] = hidden.dims();
    let mut hidden = Some(hidden);
    let mut segments = Vec::with_capacity(3);
    if pad_left > 0 {
        segments.push(
            hidden
                .as_ref()
                .expect("conv input should be present while building left padding")
                .clone()
                .slice([0..batch, 0..channels, 0..1])
                .repeat_dim(2, pad_left),
        );
    }
    if pad_right > 0 {
        segments.push(
            hidden
                .as_ref()
                .expect("conv input should be present while keeping the center segment")
                .clone(),
        );
        segments.push(
            hidden
                .take()
                .expect("conv input should be present while building right padding")
                .slice([0..batch, 0..channels, time - 1..time])
                .repeat_dim(2, pad_right),
        );
    } else {
        segments.push(
            hidden
                .take()
                .expect("conv input should be present while keeping the center segment"),
        );
    }
    Tensor::cat(segments, 2)
}
