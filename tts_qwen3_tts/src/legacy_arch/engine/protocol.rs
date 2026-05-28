use crate::error::QwenTtsInferenceError;
use crate::profile::compile::SemanticRequestCondition;

#[derive(Debug, Clone)]
pub(crate) struct PreparedCondition {
    release_label: &'static str,
    semantic: SemanticRequestCondition,
}

impl PreparedCondition {
    pub(crate) fn new(
        release_label: &'static str,
        semantic: SemanticRequestCondition,
    ) -> Result<Self, QwenTtsInferenceError> {
        if semantic.text_token_ids.is_empty() {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "prepared condition token sequence must be non-empty".to_string(),
            });
        }
        Ok(Self {
            release_label,
            semantic,
        })
    }

    pub(crate) fn release_label(&self) -> &'static str {
        self.release_label
    }

    pub(crate) fn batch_size(&self) -> usize {
        1
    }

    #[cfg(test)]
    pub(crate) fn sequence_len_hint(&self) -> usize {
        self.semantic.text_token_ids.len()
    }

    pub(crate) fn semantic(&self) -> &SemanticRequestCondition {
        &self.semantic
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CodecTokenSequence {
    batch_size: usize,
    num_code_groups: usize,
    time_steps: usize,
    token_ids: Vec<i32>,
}

impl CodecTokenSequence {
    pub(crate) fn new(
        token_ids: Vec<i32>,
        batch_size: usize,
        num_code_groups: usize,
        time_steps: usize,
    ) -> Result<Self, QwenTtsInferenceError> {
        if batch_size == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence batch size must be non-zero".to_string(),
            });
        }
        if num_code_groups == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence quantizer count must be non-zero".to_string(),
            });
        }
        if time_steps == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence time dimension must be non-zero".to_string(),
            });
        }
        let expected = batch_size
            .checked_mul(num_code_groups)
            .and_then(|n| n.checked_mul(time_steps))
            .ok_or_else(|| QwenTtsInferenceError::InvalidInput {
                message: "codec token sequence shape overflow".to_string(),
            })?;
        if token_ids.len() != expected {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "codec token sequence element mismatch: expected {expected}, got {}",
                    token_ids.len()
                ),
            });
        }
        Ok(Self {
            batch_size,
            num_code_groups,
            time_steps,
            token_ids,
        })
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    pub(crate) fn time_steps(&self) -> usize {
        self.time_steps
    }

    pub(crate) fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Waveform {
    sample_rate: u32,
    batch_size: usize,
    channels: usize,
    samples: Vec<f32>,
}

impl Waveform {
    pub(crate) fn new(
        sample_rate: u32,
        batch_size: usize,
        channels: usize,
        samples: Vec<f32>,
    ) -> Result<Self, QwenTtsInferenceError> {
        if sample_rate == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "waveform sample rate must be non-zero".to_string(),
            });
        }
        if batch_size == 0 || channels == 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform batch/channels must be non-zero, got batch={batch_size}, channels={channels}"
                ),
            });
        }
        if samples.is_empty() {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: "waveform sample payload must be non-empty".to_string(),
            });
        }
        if samples.len() % (batch_size * channels) != 0 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "waveform element mismatch: {} samples do not fit batch={batch_size}, channels={channels}",
                    samples.len()
                ),
            });
        }
        Ok(Self {
            sample_rate,
            batch_size,
            channels,
            samples,
        })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn samples(&self) -> &[f32] {
        &self.samples
    }
}
