use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::error::QwenTtsInferenceError;
use crate::profile::compile::CompiledRequest;

#[derive(Debug)]
pub(crate) struct PreparedCondition<B: Backend> {
    release_label: &'static str,
    compiled: CompiledRequest<B>,
}

impl<B: Backend> PreparedCondition<B> {
    pub(crate) fn new(
        release_label: &'static str,
        compiled: CompiledRequest<B>,
    ) -> Result<Self, QwenTtsInferenceError> {
        let [batch_size, seq_len, _hidden] = compiled.inputs_embeds.dims();
        if batch_size == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "prepared condition batch size must be non-zero".to_string(),
            });
        }
        if seq_len == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "prepared condition sequence length must be non-zero".to_string(),
            });
        }
        Ok(Self {
            release_label,
            compiled,
        })
    }

    pub(crate) fn release_label(&self) -> &'static str {
        self.release_label
    }

    pub(crate) fn dims(&self) -> [usize; 3] {
        self.compiled.inputs_embeds.dims()
    }

    pub(crate) fn into_compiled(self) -> CompiledRequest<B> {
        self.compiled
    }
}

#[derive(Debug)]
pub(crate) struct CodecTokenSequence<B: Backend> {
    token_ids: Tensor<B, 3, Int>,
}

impl<B: Backend> CodecTokenSequence<B> {
    pub(crate) fn new(
        token_ids: Tensor<B, 3, Int>,
        num_code_groups: usize,
    ) -> Result<Self, QwenTtsInferenceError> {
        let [batch_size, quantizers, time_steps] = token_ids.dims();
        if batch_size == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence batch size must be non-zero".to_string(),
            });
        }
        if quantizers != num_code_groups {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "codec token sequence quantizer mismatch: expected {num_code_groups}, got {quantizers}"
                ),
            });
        }
        if time_steps == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence time dimension must be non-zero".to_string(),
            });
        }
        Ok(Self { token_ids })
    }

    pub(crate) fn dims(&self) -> [usize; 3] {
        self.token_ids.dims()
    }

    pub(crate) fn into_tensor(self) -> Tensor<B, 3, Int> {
        self.token_ids
    }
}

#[derive(Debug)]
pub(crate) struct Waveform<B: Backend> {
    sample_rate: u32,
    samples: Tensor<B, 3>,
}

impl<B: Backend> Waveform<B> {
    pub(crate) fn new(
        sample_rate: u32,
        samples: Tensor<B, 3>,
    ) -> Result<Self, QwenTtsInferenceError> {
        let [batch_size, channels, time_steps] = samples.dims();
        if sample_rate == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "waveform sample rate must be non-zero".to_string(),
            });
        }
        if batch_size == 0 || channels == 0 || time_steps == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform dims must be non-zero, got batch={batch_size}, channels={channels}, time={time_steps}"
                ),
            });
        }
        Ok(Self { sample_rate, samples })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn samples(&self) -> &Tensor<B, 3> {
        &self.samples
    }
}
