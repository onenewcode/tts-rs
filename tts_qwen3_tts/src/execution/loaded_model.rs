use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::compiler::Qwen3TtsRequestCompiler;
use super::compiler::session_seed::{SessionSeed, materialize_session_seed};
use super::run::LoadedModel;
use super::session::{ModelSession, SessionStep};
use crate::model::codec::infer::Waveform;
use crate::model::codec::weights::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};
use crate::model::talker::infer::TalkerGenerator;
use crate::model::talker::infer::sampling::SamplingConfig as RuntimeSamplingConfig;
use crate::model::talker::weights::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};
use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsInferenceError, Qwen3TtsLoadError, Qwen3TtsPackage,
    Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, Qwen3TtsVoiceClonePrompt, QwenRequest,
};

#[cfg(not(any(
    feature = "flex",
    feature = "wgpu",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
)))]
compile_error!("enable one backend feature for tts_qwen3_tts");

#[cfg(feature = "flex")]
type RuntimeBackend = burn::backend::Flex;
#[cfg(feature = "wgpu")]
type RuntimeBackend = burn::backend::Wgpu;
#[cfg(feature = "cuda")]
type RuntimeBackend = burn::backend::Cuda;
#[cfg(feature = "rocm")]
type RuntimeBackend = burn::backend::Rocm;
#[cfg(feature = "metal")]
type RuntimeBackend = burn::backend::Metal;
#[cfg(feature = "vulkan")]
type RuntimeBackend = burn::backend::Vulkan;
#[cfg(feature = "webgpu")]
type RuntimeBackend = burn::backend::WebGpu;

#[derive(Debug)]
pub(crate) struct Qwen3TtsModelInner<B: Backend> {
    pub(crate) device: B::Device,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
    pub(crate) talker: LoadedQwen3TtsTalker<B>,
    pub(crate) decoder: LoadedQwen3TtsAudioCodec<B>,
    pub(crate) speaker_encoder: Option<crate::model::speaker::LoadedQwen3TtsSpeakerEncoder<B>>,
}

impl<B> Qwen3TtsModelInner<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn load(
        package: Qwen3TtsPackage,
        _profiling: &Qwen3TtsProfilingConfig,
        compiler: Qwen3TtsRequestCompiler,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsLoadError> {
        let talker = load_qwen3_tts_talker_for_inference::<B>(
            &package.talker_config_path,
            &package.talker_weights_path,
            device,
        )?;
        let speaker_encoder = crate::model::speaker::LoadedQwen3TtsSpeakerEncoder::load(
            &package.talker_config_path,
            &package.talker_weights_path,
            device,
        )?;
        let decoder = load_qwen3_tts_audio_codec::<B>(
            &package.codec_config_path,
            &package.codec_weights_path,
            device,
        )?;
        Ok(Self {
            device: device.clone(),
            compiler,
            talker,
            decoder,
            speaker_encoder,
        })
    }

    fn compile_session_seed(
        &self,
        request: QwenRequest,
    ) -> Result<SessionSeed<B>, Qwen3TtsInferenceError> {
        let condition = self.compiler.compile_request(&request)?;
        materialize_session_seed(
            &condition,
            &self.talker.config,
            &self.talker,
            &self.decoder,
            self.speaker_encoder.as_ref(),
            &self.device,
        )
    }

    fn start_generator(
        &self,
        seed: SessionSeed<B>,
        options: Qwen3TtsRunOptions,
    ) -> Result<TalkerGenerator<B>, Qwen3TtsInferenceError> {
        let sampling = resolve_sampling(options.sampling.as_ref(), &seed.sampling);
        TalkerGenerator::start(
            &self.talker.config,
            &self.talker,
            &seed,
            sampling,
            options.max_new_tokens.unwrap_or(seed.max_new_tokens),
            Some(seed.codec_eos_token_id),
            seed.suppress_token_ids.clone(),
        )
    }

    fn finalize_audio(
        &self,
        run: &TalkerGenerator<B>,
        reference_codec_prefix: Option<&Tensor<B, 3, Int>>,
        reference_codec_frame_count: usize,
    ) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        let generated = run.finalize()?;
        let waveform = if let Some(reference_codec_prefix) = reference_codec_prefix {
            let [batch_size, num_quantizers, time_steps] = generated.codec_token_ids.dims();
            let [prefix_batch, prefix_quantizers, prefix_steps] = reference_codec_prefix.dims();
            if prefix_batch != batch_size || prefix_quantizers != num_quantizers {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "reference codec prefix shape mismatch: expected [{batch_size}, {num_quantizers}, T], got [{prefix_batch}, {prefix_quantizers}, {prefix_steps}]"
                    ),
                });
            }
            let combined_steps = time_steps + reference_codec_frame_count;
            let codec_ids = Tensor::cat(
                vec![reference_codec_prefix.clone(), generated.codec_token_ids],
                2,
            );
            let mut waveform = self.decoder.decode_waveform(codec_ids)?;
            let total_samples = waveform.dims()[2];
            let cut_samples = reference_codec_frame_count * total_samples / combined_steps.max(1);
            waveform = waveform.slice([0..1, 0..1, cut_samples.min(total_samples)..total_samples]);
            waveform
        } else {
            self.decoder.decode_waveform(generated.codec_token_ids)?
        };
        let waveform = Waveform::from_tensor(
            u32::try_from(self.decoder.config.output_sample_rate).map_err(|_| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "decoder output sample rate {} exceeds the supported u32 audio range",
                        self.decoder.config.output_sample_rate
                    ),
                }
            })?,
            waveform,
        )?;
        let pcm = waveform.to_pcm();
        Ok(tts_infer::PcmAudio {
            pcm_i16: pcm,
            sample_rate: waveform.sample_rate(),
            channels: 1,
        })
    }

    fn create_voice_clone_prompt(
        &self,
        reference: &BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError> {
        let speaker_encoder =
            self.speaker_encoder
                .as_ref()
                .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                    message:
                        "voice clone prompt requires a Base model with speaker_encoder weights"
                            .to_string(),
                })?;
        crate::execution::conditioning::create_voice_clone_prompt(
            &self.decoder,
            speaker_encoder,
            &self.device,
            reference,
        )
    }
}

pub(crate) trait LoadedModelOps: Send + Sync {
    fn create_voice_clone_prompt(
        &self,
        reference: &BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError>;
    fn supports_voice_clone(&self) -> bool;
    fn start_session(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<Box<dyn SessionOps>, Qwen3TtsInferenceError>;
}

pub(crate) struct BackendRuntime<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
}

impl<B> BackendRuntime<B>
where
    B: Backend,
{
    pub(crate) fn new(inner: Qwen3TtsModelInner<B>) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<B> LoadedModelOps for BackendRuntime<B>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Clone + Send + Sync + 'static,
{
    fn create_voice_clone_prompt(
        &self,
        reference: &BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError> {
        self.inner.create_voice_clone_prompt(reference)
    }

    fn supports_voice_clone(&self) -> bool {
        self.inner.speaker_encoder.is_some()
    }

    fn start_session(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<Box<dyn SessionOps>, Qwen3TtsInferenceError> {
        Ok(Box::new(start_backend_session(
            &self.inner,
            request,
            options,
        )?))
    }
}

pub(crate) trait SessionOps: Send {
    fn step(&mut self) -> Result<SessionStep, Qwen3TtsInferenceError>;
    fn finish(self: Box<Self>) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError>;
}

pub(crate) struct Qwen3TtsLoadedModel {
    inner: Arc<dyn LoadedModelOps>,
}

impl std::fmt::Debug for Qwen3TtsLoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Qwen3TtsLoadedModel(..)")
    }
}

impl Qwen3TtsLoadedModel {
    pub(crate) fn load(
        package: Qwen3TtsPackage,
        profiling: &Qwen3TtsProfilingConfig,
        compiler: Qwen3TtsRequestCompiler,
    ) -> Result<Self, Qwen3TtsLoadError> {
        let device = Default::default();
        Ok(Self {
            inner: Arc::new(BackendRuntime::new(
                Qwen3TtsModelInner::<RuntimeBackend>::load(package, profiling, compiler, &device)?,
            )),
        })
    }

    pub(crate) fn create_voice_clone_prompt(
        &self,
        reference: &BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError> {
        self.inner.create_voice_clone_prompt(reference)
    }

    pub(crate) fn supports_voice_clone(&self) -> bool {
        self.inner.supports_voice_clone()
    }
}

impl Clone for Qwen3TtsLoadedModel {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl LoadedModel for Qwen3TtsLoadedModel {
    type Request = QwenRequest;
    type RunOptions = Qwen3TtsRunOptions;
    type Session = Qwen3TtsSession;
    type Error = Qwen3TtsInferenceError;

    fn start_session(
        &self,
        request: Self::Request,
        options: Self::RunOptions,
    ) -> Result<Self::Session, Self::Error> {
        Ok(Qwen3TtsSession {
            inner: self.inner.start_session(request, options)?,
        })
    }
}

pub(crate) struct Qwen3TtsSession {
    inner: Box<dyn SessionOps>,
}

impl std::fmt::Debug for Qwen3TtsSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Qwen3TtsSession(..)")
    }
}

impl ModelSession for Qwen3TtsSession {
    type Error = Qwen3TtsInferenceError;

    fn step(&mut self) -> Result<SessionStep, Self::Error> {
        self.inner.step()
    }

    fn finish(self) -> Result<tts_infer::PcmAudio, Self::Error> {
        self.inner.finish()
    }
}

#[derive(Debug)]
struct SessionImpl<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
    run: TalkerGenerator<B>,
    reference_codec_prefix: Option<Tensor<B, 3, Int>>,
    reference_codec_frame_count: usize,
}

impl<B> SessionOps for SessionImpl<B>
where
    B: Backend + Send + 'static,
    B::Device: Clone + Send + 'static,
{
    fn step(&mut self) -> Result<SessionStep, Qwen3TtsInferenceError> {
        let step_result = self.run.step(&self.inner.talker)?;
        match step_result {
            Some(step) if step.finished => Ok(SessionStep::Finished),
            Some(_) => Ok(SessionStep::Advanced),
            None => Ok(SessionStep::Finished),
        }
    }

    fn finish(self: Box<Self>) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        self.inner.finalize_audio(
            &self.run,
            self.reference_codec_prefix.as_ref(),
            self.reference_codec_frame_count,
        )
    }
}

fn start_backend_session<B>(
    inner: &Arc<Qwen3TtsModelInner<B>>,
    request: QwenRequest,
    options: Qwen3TtsRunOptions,
) -> Result<SessionImpl<B>, Qwen3TtsInferenceError>
where
    B: Backend,
    B::Device: Clone,
{
    let inner = Arc::clone(inner);
    let seed = inner.compile_session_seed(request)?;
    let reference_codec_prefix = seed.reference_codec_prefix.clone();
    let reference_codec_frame_count = seed.reference_codec_frame_count;
    let run = inner.start_generator(seed, options)?;
    Ok(SessionImpl {
        inner,
        run,
        reference_codec_prefix,
        reference_codec_frame_count,
    })
}

fn map_sampling(sampling: &crate::SamplingConfig) -> RuntimeSamplingConfig {
    RuntimeSamplingConfig {
        do_sample: sampling.do_sample,
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        repetition_penalty: sampling.repetition_penalty,
    }
}

fn resolve_sampling(
    requested: Option<&crate::SamplingOverride>,
    model_default: &crate::SamplingConfig,
) -> RuntimeSamplingConfig {
    match requested {
        None => map_sampling(model_default),
        Some(crate::SamplingOverride::Literal(config)) => map_sampling(config),
        Some(crate::SamplingOverride::GreedyFromModelDefaults) => {
            map_sampling(&crate::SamplingConfig {
                do_sample: false,
                temperature: model_default.temperature,
                top_k: model_default.top_k,
                top_p: model_default.top_p,
                seed: model_default.seed,
                repetition_penalty: model_default.repetition_penalty,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_sampling;

    #[test]
    fn explicit_sampling_config_is_used_literally() {
        let model_default = crate::SamplingConfig {
            do_sample: true,
            temperature: 0.7,
            top_k: Some(32),
            top_p: 0.85,
            seed: Some(7),
            repetition_penalty: Some(1.2),
        };
        let explicit = crate::SamplingOverride::Literal(crate::SamplingConfig::greedy());

        let runtime = resolve_sampling(Some(&explicit), &model_default);

        assert!(!runtime.do_sample);
        assert_eq!(runtime.temperature, 1.0);
        assert_eq!(runtime.top_k, None);
        assert_eq!(runtime.top_p, 1.0);
        assert_eq!(runtime.repetition_penalty, None);
    }

    #[test]
    fn greedy_override_keeps_model_penalty_defaults() {
        let model_default = crate::SamplingConfig {
            do_sample: true,
            temperature: 0.7,
            top_k: Some(32),
            top_p: 0.85,
            seed: Some(7),
            repetition_penalty: Some(1.2),
        };
        let runtime = resolve_sampling(
            Some(&crate::SamplingOverride::GreedyFromModelDefaults),
            &model_default,
        );

        assert!(!runtime.do_sample);
        assert_eq!(runtime.temperature, 0.7);
        assert_eq!(runtime.top_k, Some(32));
        assert_eq!(runtime.top_p, 0.85);
        assert_eq!(runtime.repetition_penalty, Some(1.2));
    }
}
