use burn::tensor::backend::Backend;

use crate::error::QwenTtsInferenceError;
use crate::execution::compiler::session_seed::{SessionSeed, materialize_session_seed};
use crate::model::graph::engine::components::generator::import::config::Qwen3TtsTalkerConfig;
use crate::model::graph::engine::components::generator::weights::LoadedQwen3TtsTalker;

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
        prepared: &crate::execution::compiler::SemanticRequestCondition,
        talker_config: &Qwen3TtsTalkerConfig,
        talker: &LoadedQwen3TtsTalker<B>,
        device: &B::Device,
    ) -> Result<GeneratorExecutionForm<B>, QwenTtsInferenceError> {
        let compiled = materialize_session_seed(prepared, talker_config, talker, device)?;
        Ok(GeneratorExecutionForm { compiled })
    }
}
