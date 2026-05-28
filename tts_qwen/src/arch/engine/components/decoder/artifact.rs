use std::path::Path;

use burn::tensor::backend::Backend;

use crate::arch::engine::protocol::{CodecTokenSequence, Waveform};
use crate::error::{Qwen3TtsLoadError, QwenTtsInferenceError};

use super::lowering::{DecoderExecutionForm, DecoderLowering};
use super::spec::decoder_component_spec;
use super::weights::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};

#[derive(Debug)]
pub(crate) struct DecoderArtifact<B: Backend> {
    spec: &'static crate::arch::engine::spec::ComponentSpec,
    loaded: LoadedQwen3TtsAudioCodec<B>,
}

impl<B: Backend> DecoderArtifact<B> {
    pub(crate) fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsLoadError> {
        Ok(Self {
            spec: decoder_component_spec(),
            loaded: load_qwen3_tts_audio_codec::<B>(model_dir, device)?,
        })
    }

    pub(crate) fn component_spec(&self) -> &'static crate::arch::engine::spec::ComponentSpec {
        self.spec
    }

    pub(crate) fn num_quantizers(&self) -> usize {
        self.loaded.config.decoder_config.num_quantizers
    }

    pub(crate) fn decode(
        &self,
        sequence: &CodecTokenSequence,
        device: &B::Device,
    ) -> Result<Waveform, QwenTtsInferenceError> {
        let execution = DecoderLowering::lower(sequence, device)?;
        self.decode_from_execution(execution)
    }

    fn decode_from_execution(
        &self,
        execution: DecoderExecutionForm<B>,
    ) -> Result<Waveform, QwenTtsInferenceError> {
        let waveform = super::graph::decode_waveform(&self.loaded, execution.into_tensor())?;
        DecoderLowering::lift_output(self.loaded.config.output_sample_rate as u32, waveform)
    }
}
