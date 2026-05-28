use std::path::Path;

use burn::tensor::backend::Backend;
use crate::runtime::sampling::SamplingConfig;

use crate::arch::engine::components::decoder::graph::waveform_to_pcm;
use crate::arch::engine::components::generator::graph::runner::TalkerGenerator;
use crate::error::{QwenTtsError, QwenTtsInferenceError};
use crate::profiling::with_session_context;
use crate::profile::QwenRequest;
use crate::releases::QwenReleaseManifest;
use crate::runtime::types::EngineConfig;

use super::compiler::{CompilerArtifact, ConditionCompiler};
use super::components::{decoder::artifact::DecoderArtifact, generator::artifact::GeneratorArtifact};
use super::spec::{ComponentKind, ComponentSpec, EngineSpec, qwen_engine_spec};

#[derive(Debug, Clone)]
pub(crate) struct QwenRunConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenRunStep {
    pub generated_steps: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FinishedInference {
    pub sample_rate: u32,
    pub waveform_pcm: Vec<i16>,
}

#[derive(Debug)]
pub(crate) struct QwenRun<B: Backend> {
    id: usize,
    generator_run: TalkerGenerator<B>,
}

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

    pub(crate) fn start_run(
        &self,
        request: QwenRequest,
        config: QwenRunConfig,
        device: &B::Device,
    ) -> Result<QwenRun<B>, QwenTtsError> {
        let prepared = self.compiler.prepare(&request)?;
        let execution = self.generator.execution_form(&prepared, device)?;
        let generation = self.generator.start_run(
            execution,
            config.sampling,
            config.max_new_tokens,
            Some(self.compiler.generation_config().codec_eos_token_id),
            self.compiler.generation_config().suppress_token_ids.clone(),
        )?;
        Ok(QwenRun {
            id: 0,
            generator_run: generation,
        })
    }

    pub(crate) fn step_run(&self, run: &mut QwenRun<B>) -> Result<QwenRunStep, QwenTtsError> {
        let step_idx = run.generator_run.step_idx();
        let step_result = with_session_context(run.id, step_idx, || {
            run.generator_run.step(self.generator.loaded_talker())
        })?;
        match step_result {
            Some(step) => Ok(QwenRunStep {
                generated_steps: 1,
                finished: step.finished,
            }),
            None => Ok(QwenRunStep {
                generated_steps: 0,
                finished: true,
            }),
        }
    }

    pub(crate) fn snapshot_audio(
        &self,
        run: &QwenRun<B>,
        device: &B::Device,
    ) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_generator(&run.generator_run, device)
    }

    pub(crate) fn finish_run(
        &self,
        run: QwenRun<B>,
        device: &B::Device,
    ) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_generator(&run.generator_run, device)
    }

    fn decode_finished_generator(
        &self,
        generator_run: &TalkerGenerator<B>,
        device: &B::Device,
    ) -> Result<FinishedInference, QwenTtsError> {
        let sequence = self.generator.finalize_sequence(generator_run)?;
        let waveform = self.decoder.decode(&sequence, device)?;
        let pcm = waveform_to_pcm(&waveform)?;
        Ok(FinishedInference {
            sample_rate: waveform.sample_rate(),
            waveform_pcm: pcm,
        })
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
