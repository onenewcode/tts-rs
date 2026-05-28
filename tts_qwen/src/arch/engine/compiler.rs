use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use tokenizers::Tokenizer;

use crate::error::QwenTtsInferenceError;
use crate::io::tokenizer::load_qwen3_tts_tokenizer;
use crate::profile::QwenRequest;
use crate::profile::compile::compile_request;
use crate::profile::model_config::GenerationConfig;
use crate::releases::QwenReleaseManifest;

use super::components::generator::artifact::GeneratorArtifact;
use super::protocol::PreparedCondition;

#[derive(Debug, Clone)]
pub(crate) struct CompilerArtifact {
    model_dir: PathBuf,
    release: &'static QwenReleaseManifest,
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct ConditionCompiler {
    artifact: CompilerArtifact,
}

impl ConditionCompiler {
    pub(crate) fn load(
        model_dir: impl AsRef<Path>,
        release: &'static QwenReleaseManifest,
    ) -> Result<Self, QwenTtsInferenceError> {
        let model_dir = model_dir.as_ref().to_path_buf();
        tracing::debug!(architecture = release.architecture.label, ?release.architecture.id, model_dir = %model_dir.display(), "loading condition compiler");
        let tokenizer = load_qwen3_tts_tokenizer(&model_dir)?;
        let generation_config = (release.architecture.load_generation_config)(&model_dir, release.profile)?;
        Ok(Self {
            artifact: CompilerArtifact {
                model_dir,
                release,
                tokenizer,
                generation_config,
            },
        })
    }


    pub(crate) fn into_artifact(self) -> CompilerArtifact {
        self.artifact
    }
}

impl CompilerArtifact {
    pub(crate) fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }

    pub(crate) fn prepare<B: Backend>(
        &self,
        generator: &GeneratorArtifact<B>,
        request: &QwenRequest,
        device: &B::Device,
    ) -> Result<PreparedCondition<B>, QwenTtsInferenceError> {
        let compiled = compile_request(
            self.release,
            &self.tokenizer,
            &self.model_dir,
            generator.talker_config(),
            generator.loaded_talker(),
            request,
            device,
        )?;
        PreparedCondition::new(self.release.label, compiled)
    }
}
