// Audio codec causal convolution wrappers around Burn conv modules.
use burn::module::Module;
use burn::nn::conv::{Conv1d, ConvTranspose1d};
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct AudioCodecCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
    #[module(skip)]
    pub(crate) pad_mode: PadMode,
}

#[derive(Module, Debug)]
pub struct AudioCodecCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
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
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let (padding_total, extra_padding) = conv1d_padding(&self.conv, hidden.dims()[2]);

        // Burn pad creates its temporary in the backend default float dtype, so pad_1d enters
        // that dtype only for the pad op and restores the caller's runtime dtype afterward.
        self.conv
            .forward(pad_1d(hidden, padding_total, extra_padding, self.pad_mode))
    }
}

pub(crate) fn conv1d_padding<B: Backend>(conv: &Conv1d<B>, len: usize) -> (usize, usize) {
    let effective_kernel = (conv.kernel_size - 1) * conv.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(conv.stride);
    let extra_padding = extra_padding_for_conv1d(len, effective_kernel, conv.stride, padding_total);
    (padding_total, extra_padding)
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

/// Pad the time axis of a `[batch, channels, time]` tensor through Burn's pad kernel while
/// preserving the caller's runtime dtype.
///
/// Burn currently materializes the pad temporary in the backend default float dtype, so this
/// helper enters that dtype for the pad op and casts back before returning.
pub(crate) fn pad_1d<B: Backend>(
    hidden: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
    pad_mode: PadMode,
) -> Tensor<B, 3> {
    let hidden_dtype = hidden.dtype();
    let stable_dtype = Tensor::<B, 1>::zeros([1], &hidden.device()).dtype();
    // Enter Burn's default float lane only for pad, then restore the caller's runtime dtype.
    let hidden = hidden.dequantize().cast(stable_dtype);
    let hidden = hidden.pad([(0, 0), (0, 0), (pad_left, pad_right)], pad_mode);
    hidden.cast(hidden_dtype)
}

#[cfg(test)]
mod tests {
    use burn::tensor::DType;
    use burn::tensor::ops::PadMode;

    use crate::loading::runtime::RuntimeBackend;

    use super::pad_1d;

    #[test]
    fn constant_padding_preserves_runtime_dtype() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let hidden =
                burn::tensor::Tensor::<RuntimeBackend, 1>::from_data([1.0f32, -2.0], &device)
                    .reshape([1, 1, 2])
                    .cast(dtype);

            let padded = pad_1d(hidden, 1, 2, PadMode::Constant(0.0));

            assert_eq!(padded.dtype(), dtype);
            assert_eq!(padded.dims(), [1, 1, 5]);

            let samples = padded
                .try_into_data()
                .expect("padded tensor should be readable")
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("padded tensor should convert to f32");
            assert_eq!(samples, vec![0.0, 1.0, -2.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn edge_padding_preserves_runtime_dtype() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let hidden =
                burn::tensor::Tensor::<RuntimeBackend, 1>::from_data([1.0f32, -2.0], &device)
                    .reshape([1, 1, 2])
                    .cast(dtype);

            let padded = pad_1d(hidden, 2, 1, PadMode::Edge);

            assert_eq!(padded.dtype(), dtype);
            assert_eq!(padded.dims(), [1, 1, 5]);

            let samples = padded
                .try_into_data()
                .expect("padded tensor should be readable")
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("padded tensor should convert to f32");
            assert_eq!(samples, vec![1.0, 1.0, 1.0, -2.0, -2.0]);
        }
    }
}
