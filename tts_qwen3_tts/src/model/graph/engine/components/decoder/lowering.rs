use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::error::QwenTtsInferenceError;

use super::graph::Waveform;

#[derive(Debug)]
pub(crate) struct DecoderExecutionForm<B: Backend> {
    token_ids: Tensor<B, 3, burn::tensor::Int>,
}

impl<B: Backend> DecoderExecutionForm<B> {
    #[cfg(test)]
    pub(crate) fn batch_size(&self) -> usize {
        self.token_ids.dims()[0]
    }

    #[cfg(test)]
    pub(crate) fn num_quantizers(&self) -> usize {
        self.token_ids.dims()[1]
    }

    #[cfg(test)]
    pub(crate) fn time_steps(&self) -> usize {
        self.token_ids.dims()[2]
    }

    pub(crate) fn into_tensor(self) -> Tensor<B, 3, burn::tensor::Int> {
        self.token_ids
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DecoderLowering;

impl DecoderLowering {
    pub(crate) fn lower<B: Backend>(
        token_ids: Vec<i32>,
        batch_size: usize,
        num_quantizers: usize,
        time_steps: usize,
        device: &B::Device,
    ) -> Result<DecoderExecutionForm<B>, QwenTtsInferenceError> {
        if batch_size == 0 || num_quantizers == 0 || time_steps == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "decoder lowering requires finalized codec token sequences".to_string(),
            });
        }
        let token_ids = Tensor::<B, 3, burn::tensor::Int>::from_data(
            TensorData::new(
                token_ids,
                [batch_size, num_quantizers, time_steps],
            ),
            device,
        );
        Ok(DecoderExecutionForm { token_ids })
    }

    pub(crate) fn lift_output<B: Backend>(
        sample_rate: u32,
        waveform: Tensor<B, 3>,
    ) -> Result<Waveform, QwenTtsInferenceError> {
        let [batch_size, channels, _time_steps] = waveform.dims();
        let samples = waveform
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|e| QwenTtsInferenceError::TensorRead {
                message: format!("failed to read waveform: {e}"),
            })?;
        Waveform::new(sample_rate, batch_size, channels, samples)
    }
}
