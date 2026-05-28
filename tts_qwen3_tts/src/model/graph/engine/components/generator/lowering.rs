use burn::tensor::backend::Backend;

use crate::compiler::session_seed::{SessionSeed, materialize_session_seed};
use crate::model::graph::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::model::graph::engine::components::generator::weights::LoadedQwen3TtsTalker;
use crate::error::QwenTtsInferenceError;

use super::graph::runner::TalkerGenerationOutput;

#[derive(Debug)]
pub(crate) struct GeneratorExecutionForm<B: Backend> {
    compiled: SessionSeed<B>,
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

    pub(crate) fn into_compiled(self) -> SessionSeed<B> {
        self.compiled
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GeneratorLowering;

impl GeneratorLowering {
    pub(crate) fn lower<B: Backend>(
        prepared: &crate::compiler::SemanticRequestCondition,
        talker_config: &Qwen3TtsTalkerConfig,
        talker: &LoadedQwen3TtsTalker<B>,
        device: &B::Device,
    ) -> Result<GeneratorExecutionForm<B>, QwenTtsInferenceError> {
        let compiled = materialize_session_seed(prepared, talker_config, talker, device)?;
        Ok(GeneratorExecutionForm { compiled })
    }

    pub(crate) fn lift_output<B: Backend>(
        output: TalkerGenerationOutput<B>,
        num_code_groups: usize,
    ) -> Result<burn::tensor::TensorData, QwenTtsInferenceError> {
        let dims = output.codec_token_ids.dims();
        let [_batch_size, quantizers, time_steps] = dims;
        let token_ids = output.codec_token_ids.into_data();
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
        Ok(token_ids)
    }
}
