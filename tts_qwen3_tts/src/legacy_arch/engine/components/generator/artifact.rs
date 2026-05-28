use std::path::Path;

use burn::tensor::backend::Backend;
use crate::runtime::sampling::SamplingConfig;

use crate::arch::engine::protocol::{CodecTokenSequence, PreparedCondition};
use crate::error::{Qwen3TtsLoadError, QwenTtsInferenceError};

use super::graph::runner::TalkerGenerator;
use super::lowering::{GeneratorExecutionForm, GeneratorLowering};
use super::spec::generator_component_spec;
use super::weights::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};

#[derive(Debug)]
pub(crate) struct GeneratorArtifact<B: Backend> {
    spec: &'static crate::arch::engine::spec::ComponentSpec,
    loaded: LoadedQwen3TtsTalker<B>,
}

impl<B: Backend> GeneratorArtifact<B> {
    pub(crate) fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsLoadError> {
        Ok(Self {
            spec: generator_component_spec(),
            loaded: load_qwen3_tts_talker_for_inference::<B>(model_dir, device)?,
        })
    }

    pub(crate) fn component_spec(&self) -> &'static crate::arch::engine::spec::ComponentSpec {
        self.spec
    }


    pub(crate) fn loaded_talker(&self) -> &LoadedQwen3TtsTalker<B> {
        &self.loaded
    }

    pub(crate) fn num_code_groups(&self) -> usize {
        self.loaded.config.talker_config.num_code_groups
    }

    pub(crate) fn execution_form(
        &self,
        prepared: &PreparedCondition,
        device: &B::Device,
    ) -> Result<GeneratorExecutionForm<B>, QwenTtsInferenceError> {
        GeneratorLowering::lower(
            prepared,
            &self.loaded.config.talker_config,
            &self.loaded,
            device,
        )
    }

    pub(crate) fn start_run(
        &self,
        execution: GeneratorExecutionForm<B>,
        sampling: SamplingConfig,
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        suppress_token_ids: Vec<usize>,
    ) -> Result<TalkerGenerator<B>, QwenTtsInferenceError> {
        self.start_from_execution(
            execution,
            sampling,
            max_new_tokens,
            eos_token_id,
            suppress_token_ids,
        )
    }

    pub(crate) fn finalize_sequence(
        &self,
        run: &TalkerGenerator<B>,
    ) -> Result<CodecTokenSequence, QwenTtsInferenceError> {
        let output = run.finalize()?;
        GeneratorLowering::lift_output(output, self.num_code_groups())
    }

    fn start_from_execution(
        &self,
        execution: GeneratorExecutionForm<B>,
        sampling: SamplingConfig,
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        suppress_token_ids: Vec<usize>,
    ) -> Result<TalkerGenerator<B>, QwenTtsInferenceError> {
        let compiled = execution.into_compiled();
        TalkerGenerator::start(
            &self.loaded.config.talker_config,
            &self.loaded,
            &compiled,
            sampling,
            max_new_tokens,
            eos_token_id,
            suppress_token_ids,
        )
    }
}
