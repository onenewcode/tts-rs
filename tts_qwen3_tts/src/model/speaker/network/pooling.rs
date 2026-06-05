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

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let model_dtype = x.dtype();
        let stable_dtype = Tensor::<B, 1>::zeros([1], &x.device()).dtype();
        let [batch, channels, time] = x.dims();

        let x_sq = x.clone().powi_scalar(2);
        let mean = x.clone().mean_dim(2);
        let variance = (x_sq.clone().mean_dim(2) - mean.clone().powi_scalar(2)).clamp_min(0.0);
        let std = {
            let std = (variance.dequantize().cast(stable_dtype) + 1e-5).sqrt();
            std.cast(model_dtype)
        };

        let attn_in = Tensor::cat(
            vec![
                x.clone(),
                mean.expand([batch, channels, time]),
                std.expand([batch, channels, time]),
            ],
            1,
        );
        let attn = self.tdnn.forward(attn_in);
        let attn = {
            let attn = attn.dequantize().cast(stable_dtype).tanh();
            attn.cast(model_dtype)
        };
        let attn = self.conv.forward(attn);
        let attn = {
            let attn = softmax(attn.dequantize().cast(stable_dtype), 2);
            attn.cast(model_dtype)
        };

        let weighted_mean = (x * attn.clone()).sum_dim(2);
        let weighted_second_moment = (x_sq * attn).sum_dim(2);
        let weighted_var =
            (weighted_second_moment - weighted_mean.clone().powi_scalar(2)).clamp_min(0.0);
        let weighted_std = {
            let weighted_std = (weighted_var.dequantize().cast(stable_dtype) + 1e-5).sqrt();
            weighted_std.cast(model_dtype)
        };

        Tensor::cat(vec![weighted_mean, weighted_std], 1)
    }
}

#[cfg(test)]
mod tests {
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::backend::Backend;
    use burn::tensor::{DType, Tensor};

    use crate::loading::runtime::RuntimeBackend;

    use super::AttentiveStatisticsPooling;

    struct CastFloatParams {
        dtype: DType,
    }

    impl<B: Backend> ModuleMapper<B> for CastFloatParams {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.cast(self.dtype), mapper)
        }
    }

    #[test]
    fn pooling_forward_restores_runtime_dtype() {
        let device = Default::default();

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let pooling = AttentiveStatisticsPooling::<RuntimeBackend>::new(2, 2, &device)
                .map(&mut CastFloatParams { dtype });
            let hidden = Tensor::<RuntimeBackend, 1>::from_floats(
                [0.5, -0.5, 1.0, 0.0, 0.25, 0.75, -1.0, 0.5],
                &device,
            )
            .reshape([1, 2, 4])
            .cast(dtype);

            let pooled = pooling.forward(hidden);

            assert_eq!(pooled.dtype(), dtype);
            assert_eq!(pooled.dims(), [1, 4, 1]);
            let _ = pooled
                .try_into_data()
                .expect("pooled statistics should be readable");
        }
    }
}
