use burn::tensor::backend::Backend;

use crate::arch::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::arch::engine::components::generator::weights::LoadedQwen3TtsTalker;
use crate::arch::engine::protocol::{CodecTokenSequence, PreparedCondition};
use crate::error::QwenTtsInferenceError;
use crate::profile::compile::{CompiledRequest, materialize_compiled_request};

use super::graph::runner::TalkerGenerationOutput;

#[derive(Debug)]
pub(crate) struct GeneratorExecutionForm<B: Backend> {
    compiled: CompiledRequest<B>,
}

impl<B: Backend> GeneratorExecutionForm<B> {
    #[cfg(test)]
    pub(crate) fn batch_size(&self) -> usize {
        self.compiled.inputs_embeds.dims()[0]
    }

    #[cfg(test)]
    pub(crate) fn sequence_len(&self) -> usize {
        self.compiled.inputs_embeds.dims()[1]
    }

    pub(crate) fn into_compiled(self) -> CompiledRequest<B> {
        self.compiled
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GeneratorLowering;

impl GeneratorLowering {
    pub(crate) fn lower<B: Backend>(
        prepared: &PreparedCondition,
        talker_config: &Qwen3TtsTalkerConfig,
        talker: &LoadedQwen3TtsTalker<B>,
        device: &B::Device,
    ) -> Result<GeneratorExecutionForm<B>, QwenTtsInferenceError> {
        let release_label = prepared.release_label();
        if prepared.batch_size() != 1 {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "generator lowering currently supports exactly one request per session for `{release_label}`, got batch {}",
                    prepared.batch_size()
                ),
            });
        }
        let compiled = materialize_compiled_request(prepared.semantic(), talker_config, talker, device)?;
        Ok(GeneratorExecutionForm { compiled })
    }

    pub(crate) fn lift_output<B: Backend>(
        output: TalkerGenerationOutput<B>,
        num_code_groups: usize,
    ) -> Result<CodecTokenSequence, QwenTtsInferenceError> {
        let dims = output.codec_token_ids.dims();
        let [batch_size, quantizers, time_steps] = dims;
        let token_ids = output
            .codec_token_ids
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .map_err(|e| QwenTtsInferenceError::TensorRead {
                message: format!("failed to read codec token sequence: {e}"),
            })?;
        if quantizers != num_code_groups {
            return Err(QwenTtsInferenceError::InvalidInput {
                message: format!(
                    "codec token sequence quantizer mismatch: expected {num_code_groups}, got {quantizers}"
                ),
            });
        }
        CodecTokenSequence::new(token_ids, batch_size, quantizers, time_steps)
    }
}
