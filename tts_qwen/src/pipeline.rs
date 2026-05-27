use std::path::Path;

use burn::tensor::backend::Backend;

use crate::adapter::{
    Qwen3TtsInferOutput, Qwen3TtsPipelineError, Qwen3TtsPipelineLoadReport, QwenTtsAdapter,
};
use crate::core::{LocalInferenceCore, LocalInferenceOptions, LocalInferenceRun};
use crate::frontend::{
    CustomVoiceBatch, CustomVoiceRequest, FrontendOutput, Qwen3TtsTextTokenizer,
};
use crate::{CustomVoiceGenerationConfig, Qwen3TtsAudioCodecConfig, Qwen3TtsTalkerConfig};

pub type Qwen3TtsInferOptions = LocalInferenceOptions;

#[derive(Debug)]
pub struct Qwen3TtsPipeline<B>
where
    B: Backend,
    B::Device: Clone,
{
    core: LocalInferenceCore<B, QwenTtsAdapter<B>>,
}

impl<B> Qwen3TtsPipeline<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub fn load(
        model_dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsPipelineError> {
        let core = LocalInferenceCore::<B, QwenTtsAdapter<B>>::load(model_dir, device)?;
        Ok(Self { core })
    }

    pub fn core(&self) -> &LocalInferenceCore<B, QwenTtsAdapter<B>> {
        &self.core
    }

    pub fn model_dir(&self) -> &Path {
        self.core.adapter().model_dir()
    }

    pub fn text_tokenizer(&self) -> &Qwen3TtsTextTokenizer {
        self.core.adapter().text_tokenizer()
    }

    pub fn talker_config(&self) -> &Qwen3TtsTalkerConfig {
        self.core.adapter().talker_config()
    }

    pub fn audio_codec_config(&self) -> &Qwen3TtsAudioCodecConfig {
        self.core.adapter().audio_codec_config()
    }

    pub fn generation_config(&self) -> &CustomVoiceGenerationConfig {
        self.core.adapter().generation_config()
    }

    pub fn load_report(&self) -> Qwen3TtsPipelineLoadReport {
        self.core.load_report()
    }

    pub fn build_frontend(
        &self,
        request: &CustomVoiceRequest,
    ) -> Result<FrontendOutput<B>, Qwen3TtsPipelineError> {
        self.core.adapter().build_frontend(request)
    }

    pub fn build_frontend_batch(
        &self,
        batch: &CustomVoiceBatch,
    ) -> Result<FrontendOutput<B>, Qwen3TtsPipelineError> {
        self.core.adapter().build_frontend_batch(batch)
    }

    pub fn infer(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsInferOptions,
    ) -> Result<Qwen3TtsInferOutput<B>, Qwen3TtsPipelineError> {
        Ok(self.infer_with_profile(request, options)?.output)
    }

    pub fn infer_with_profile(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsInferOptions,
    ) -> Result<LocalInferenceRun<Qwen3TtsInferOutput<B>>, Qwen3TtsPipelineError> {
        self.core.infer(request, options)
    }

    pub fn infer_to_wav(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsInferOptions,
        path: impl AsRef<Path>,
    ) -> Result<Qwen3TtsInferOutput<B>, Qwen3TtsPipelineError> {
        Ok(self
            .infer_to_wav_with_profile(request, options, path)?
            .output)
    }

    pub fn infer_to_wav_with_profile(
        &self,
        request: &CustomVoiceRequest,
        options: &Qwen3TtsInferOptions,
        path: impl AsRef<Path>,
    ) -> Result<LocalInferenceRun<Qwen3TtsInferOutput<B>>, Qwen3TtsPipelineError> {
        self.core.infer_to_file(request, options, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_options_default_to_greedy_generation() {
        let options = Qwen3TtsInferOptions::default();

        assert_eq!(options.max_new_tokens, 256);
        assert!(!options.sampling.do_sample);
        assert_eq!(options.sampling.temperature, 1.0);
    }
}
