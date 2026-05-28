use burn::tensor::backend::Backend;

use crate::error::QwenTtsInferenceError;

use crate::arch::engine::protocol::{CodecTokenSequence, Waveform};

#[derive(Debug)]
pub(crate) struct DecoderExecutionForm<B: Backend> {
    sequence: CodecTokenSequence<B>,
}

impl<B: Backend> DecoderExecutionForm<B> {
    #[cfg(test)]
    pub(crate) fn batch_size(&self) -> usize {
        self.sequence.dims()[0]
    }

    #[cfg(test)]
    pub(crate) fn num_quantizers(&self) -> usize {
        self.sequence.dims()[1]
    }

    #[cfg(test)]
    pub(crate) fn time_steps(&self) -> usize {
        self.sequence.dims()[2]
    }

    pub(crate) fn into_sequence(self) -> CodecTokenSequence<B> {
        self.sequence
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DecoderLowering;

impl DecoderLowering {
    pub(crate) fn lower<B: Backend>(
        sequence: CodecTokenSequence<B>,
    ) -> Result<DecoderExecutionForm<B>, QwenTtsInferenceError> {
        if sequence.dims()[2] == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "decoder lowering requires finalized codec token sequences".to_string(),
            });
        }
        Ok(DecoderExecutionForm { sequence })
    }

    pub(crate) fn lift_output<B: Backend>(
        sample_rate: u32,
        waveform: burn::tensor::Tensor<B, 3>,
    ) -> Result<Waveform<B>, QwenTtsInferenceError> {
        Waveform::new(sample_rate, waveform)
    }
}
