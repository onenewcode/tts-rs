use burn::tensor::backend::Backend;

use crate::error::QwenTtsInferenceError;

use super::graph::runner::TalkerGenerationOutput;
use crate::arch::engine::protocol::{CodecTokenSequence, PreparedCondition};

#[derive(Debug)]
pub(crate) struct GeneratorExecutionForm<B: Backend> {
    prepared: PreparedCondition<B>,
}

impl<B: Backend> GeneratorExecutionForm<B> {
    #[cfg(test)]
    pub(crate) fn batch_size(&self) -> usize {
        self.prepared.dims()[0]
    }

    #[cfg(test)]
    pub(crate) fn sequence_len(&self) -> usize {
        self.prepared.dims()[1]
    }

    pub(crate) fn into_prepared(self) -> PreparedCondition<B> {
        self.prepared
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GeneratorLowering;

impl GeneratorLowering {
    pub(crate) fn lower<B: Backend>(
        prepared: PreparedCondition<B>,
    ) -> Result<GeneratorExecutionForm<B>, QwenTtsInferenceError> {
        let release_label = prepared.release_label();
        let dims = prepared.dims();
        if dims[0] != 1 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "generator lowering currently supports exactly one request per session for `{release_label}`, got batch {}",
                    dims[0]
                ),
            });
        }
        Ok(GeneratorExecutionForm { prepared })
    }

    pub(crate) fn lift_output<B: Backend>(
        output: TalkerGenerationOutput<B>,
        num_code_groups: usize,
    ) -> Result<CodecTokenSequence<B>, QwenTtsInferenceError> {
        CodecTokenSequence::new(output.codec_token_ids, num_code_groups)
    }
}
