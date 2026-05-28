use std::path::Path;

use burn::tensor::backend::Backend;
use tts_core::runtime::sampling::SamplingConfig;

use crate::error::QwenTtsError;
use crate::profiling::with_session_context;
use crate::profile::QwenRequest;
use crate::releases::QwenReleaseManifest;
use crate::runtime::types::EngineConfig;

use super::EngineArtifact;
use crate::arch::engine::components::decoder::graph::waveform_to_pcm;
use crate::arch::engine::components::generator::graph::runner::TalkerGenerator;

#[derive(Debug, Clone)]
pub(crate) struct QwenEngineBridge;

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
pub(crate) struct QwenEngine<B: Backend>
where
    B::Device: Clone,
{
    artifact: EngineArtifact<B>,
    device: B::Device,
}

impl QwenEngineBridge {
    pub(crate) fn load_engine<B: Backend>(
        model_dir: impl AsRef<Path>,
        release: &'static QwenReleaseManifest,
        device: &B::Device,
        config: EngineConfig,
    ) -> Result<QwenEngine<B>, QwenTtsError>
    where
        B::Device: Clone,
    {
        let artifact = EngineArtifact::assemble(model_dir.as_ref(), release, device, config)?;
        Ok(QwenEngine {
            artifact,
            device: device.clone(),
        })
    }
}

impl<B> QwenEngine<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn start_run(
        &self,
        request: QwenRequest,
        config: QwenRunConfig,
    ) -> Result<QwenRun<B>, QwenTtsError> {
        let prepared = self
            .artifact
            .compiler()
            .prepare(self.artifact.generator(), &request, &self.device)?;
        let generation = self.artifact.generator().start_run(
            prepared,
            config.sampling,
            config.max_new_tokens,
            Some(self.artifact.compiler().generation_config().codec_eos_token_id),
            self.artifact
                .compiler()
                .generation_config()
                .suppress_token_ids
                .clone(),
        )?;
        Ok(QwenRun {
            id: 0,
            generator_run: generation,
        })
    }

    pub(crate) fn step_run(&self, run: &mut QwenRun<B>) -> Result<QwenRunStep, QwenTtsError> {
        let step_idx = run.generator_run.step_idx();
        let step_result = with_session_context(run.id, step_idx, || {
            run.generator_run.step(self.artifact.generator().loaded_talker())
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

    pub(crate) fn snapshot_audio(&self, run: &QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_generator(&run.generator_run)
    }

    pub(crate) fn finish_run(&self, run: QwenRun<B>) -> Result<FinishedInference, QwenTtsError> {
        self.decode_finished_generator(&run.generator_run)
    }

    fn decode_finished_generator(
        &self,
        generator_run: &TalkerGenerator<B>,
    ) -> Result<FinishedInference, QwenTtsError> {
        let sequence = self.artifact.generator().finalize_sequence(generator_run)?;
        let waveform = self.artifact.decoder().decode(sequence)?;
        let pcm = waveform_to_pcm(waveform.samples())?;
        Ok(FinishedInference {
            sample_rate: waveform.sample_rate(),
            waveform_pcm: pcm,
        })
    }
}
