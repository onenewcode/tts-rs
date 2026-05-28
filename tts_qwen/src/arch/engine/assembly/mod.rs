pub mod bridge;

use std::path::Path;

use burn::tensor::backend::Backend;

use crate::error::{QwenTtsError, QwenTtsInferenceError};
use crate::releases::QwenReleaseManifest;
use crate::runtime::types::EngineConfig;

use super::compiler::{CompilerArtifact, ConditionCompiler};
use super::components::{decoder::artifact::DecoderArtifact, generator::artifact::GeneratorArtifact};
use super::spec::{ComponentKind, ComponentSpec, EngineSpec, qwen_engine_spec};

#[derive(Debug)]
pub(crate) struct EngineArtifact<B: Backend> {
    spec: EngineSpec,
    compiler: CompilerArtifact,
    generator: GeneratorArtifact<B>,
    decoder: DecoderArtifact<B>,
}

impl<B: Backend> EngineArtifact<B> {
    pub(crate) fn assemble(
        model_dir: impl AsRef<Path>,
        release: &'static QwenReleaseManifest,
        device: &B::Device,
        config: EngineConfig,
    ) -> Result<Self, QwenTtsError> {
        crate::profiling::configure(&config.profiling);
        let compiler = ConditionCompiler::load(&model_dir, release)?;
        let generator = GeneratorArtifact::load(&model_dir, device)?;
        let decoder = DecoderArtifact::load(model_dir, device)?;
        let artifact = Self {
            spec: qwen_engine_spec(),
            compiler: compiler.into_artifact(),
            generator,
            decoder,
        };
        artifact.validate_compatibility()?;
        Ok(artifact)
    }

    pub(crate) fn compiler(&self) -> &CompilerArtifact {
        &self.compiler
    }

    pub(crate) fn generator(&self) -> &GeneratorArtifact<B> {
        &self.generator
    }

    pub(crate) fn decoder(&self) -> &DecoderArtifact<B> {
        &self.decoder
    }

    fn validate_compatibility(&self) -> Result<(), QwenTtsInferenceError> {
        validate_engine_contract(
            &self.spec,
            self.generator.component_spec(),
            self.decoder.component_spec(),
            self.generator.num_code_groups(),
            self.decoder.num_quantizers(),
        )
    }
}

fn validate_engine_contract(
    spec: &EngineSpec,
    generator: &ComponentSpec,
    decoder: &ComponentSpec,
    generator_groups: usize,
    decoder_quantizers: usize,
) -> Result<(), QwenTtsInferenceError> {
    if spec.component(ComponentKind::AcousticGenerator).is_none()
        || spec.component(ComponentKind::AudioDecoder).is_none()
    {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!("engine spec `{}` is missing required DAG components", spec.name),
        });
    }

    if generator_groups != decoder_quantizers {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: format!(
                "engine assembly mismatch: generator emits {generator_groups} code groups but decoder expects {decoder_quantizers} quantizers"
            ),
        });
    }

    if generator.execution_boundary.produces != decoder.execution_boundary.accepts {
        return Err(QwenTtsInferenceError::InvalidInput {
            message: "engine assembly mismatch: generator/decoder boundary protocols are incompatible".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_engine_contract;
    use crate::arch::engine::spec::{ComponentKind, qwen_engine_spec};

    #[test]
    fn assembly_accepts_matching_generator_decoder_contracts() {
        let spec = qwen_engine_spec();
        let generator = spec.component(ComponentKind::AcousticGenerator).unwrap();
        let decoder = spec.component(ComponentKind::AudioDecoder).unwrap();
        validate_engine_contract(&spec, generator, decoder, 16, 16).unwrap();
    }

    #[test]
    fn assembly_rejects_quantizer_mismatches() {
        let spec = qwen_engine_spec();
        let generator = spec.component(ComponentKind::AcousticGenerator).unwrap();
        let decoder = spec.component(ComponentKind::AudioDecoder).unwrap();
        let error = validate_engine_contract(&spec, generator, decoder, 16, 12)
            .unwrap_err()
            .to_string();
        assert!(error.contains("generator emits 16 code groups but decoder expects 12 quantizers"));
    }

    #[test]
    fn assembly_rejects_protocol_boundary_mismatches() {
        let spec = qwen_engine_spec();
        let generator = spec.component(ComponentKind::AcousticGenerator).unwrap();
        let mut decoder = *spec.component(ComponentKind::AudioDecoder).unwrap();
        decoder.execution_boundary.accepts = "PreparedCondition";
        let error = validate_engine_contract(&spec, generator, &decoder, 16, 16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("boundary protocols are incompatible"));
    }
}
