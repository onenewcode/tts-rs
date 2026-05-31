use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::execution::compiler::session_seed::{materialize_session_seed, SessionSeed};
use crate::execution::compiler::Qwen3TtsRequestCompiler;
use crate::execution::run::LoadedModel;
use crate::execution::session::{ModelSession, SessionStep};
use crate::model::codec::loading::{load_qwen3_tts_audio_codec, LoadedQwen3TtsAudioCodec};
use crate::model::codec::runtime::{decode_waveform, lift_waveform, waveform_to_pcm};
use crate::model::talker::infer::TalkerGenerator;
use crate::model::talker::sampling::SamplingConfig as RuntimeSamplingConfig;
use crate::model::talker::weights::{load_qwen3_tts_talker_for_inference, LoadedQwen3TtsTalker};
use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsBackend, Qwen3TtsInferenceError, Qwen3TtsLoadError,
    Qwen3TtsPackage, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, Qwen3TtsVoiceClonePrompt,
    QwenRequest,
};

pub(crate) mod codec;
pub(crate) mod nn;
mod runtime;
pub(crate) mod speaker;
pub(crate) mod talker;

#[derive(Debug)]
pub(crate) struct Qwen3TtsModelInner<B: Backend> {
    pub(crate) device: B::Device,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
    pub(crate) talker: LoadedQwen3TtsTalker<B>,
    pub(crate) decoder: LoadedQwen3TtsAudioCodec<B>,
    pub(crate) speaker_encoder: Option<speaker::LoadedQwen3TtsSpeakerEncoder<B>>,
}

impl<B> Qwen3TtsModelInner<B>
where
    B: Backend,
    B::Device: Clone,
{
    fn load(
        package: Qwen3TtsPackage,
        profiling: &Qwen3TtsProfilingConfig,
        compiler: Qwen3TtsRequestCompiler,
        device: &B::Device,
    ) -> Result<Self, Qwen3TtsLoadError> {
        crate::execution::profiling::configure(profiling);
        let talker = load_qwen3_tts_talker_for_inference::<B>(
            &package.talker_config_path,
            &package.talker_weights_path,
            device,
        )?;
        let speaker_encoder = speaker::LoadedQwen3TtsSpeakerEncoder::load(
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
        materialize_session_seed(&condition, &self.talker.config, &self.talker, &self.device)
    }

    fn start_generator(
        &self,
        seed: SessionSeed<B>,
        options: Qwen3TtsRunOptions,
    ) -> Result<TalkerGenerator<B>, Qwen3TtsInferenceError> {
        TalkerGenerator::start(
            &self.talker.config,
            &self.talker,
            &seed,
            map_sampling(&options.sampling),
            options.max_new_tokens.unwrap_or(seed.max_new_tokens),
            Some(seed.codec_eos_token_id),
            seed.suppress_token_ids.clone(),
        )
    }

    fn finalize_audio(
        &self,
        run: &TalkerGenerator<B>,
        reference_codec_frames: Option<&[Vec<i64>]>,
    ) -> Result<tts_core::PcmAudio, Qwen3TtsInferenceError> {
        let generated = run.finalize()?;
        let waveform = if let Some(reference_codec_frames) = reference_codec_frames {
            let [batch_size, num_quantizers, time_steps] = generated.codec_token_ids.dims();
            let reference_prefix = reference_codec_prefix_tensor::<B>(
                reference_codec_frames,
                batch_size,
                num_quantizers,
                &self.device,
            )?;
            let combined_steps = time_steps + reference_codec_frames.len();
            let codec_ids = Tensor::cat(vec![reference_prefix, generated.codec_token_ids], 2);
            let mut waveform = decode_waveform(&self.decoder, codec_ids)?;
            let total_samples = waveform.dims()[2];
            let cut_samples = reference_codec_frames.len() * total_samples / combined_steps.max(1);
            waveform = waveform.slice([0..1, 0..1, cut_samples.min(total_samples)..total_samples]);
            waveform
        } else {
            decode_waveform(&self.decoder, generated.codec_token_ids)?
        };
        let waveform = lift_waveform(self.decoder.config.output_sample_rate as u32, waveform)?;
        let pcm = waveform_to_pcm(&waveform)?;
        Ok(tts_core::PcmAudio {
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

trait LoadedModelOps: Send + Sync {
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

struct BackendRuntime<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
}

impl<B> BackendRuntime<B>
where
    B: Backend,
{
    fn new(inner: Qwen3TtsModelInner<B>) -> Self {
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

trait SessionOps: Send {
    fn step(&mut self) -> Result<SessionStep, Qwen3TtsInferenceError>;
    fn finish(self: Box<Self>) -> Result<tts_core::PcmAudio, Qwen3TtsInferenceError>;
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
        backend: Qwen3TtsBackend,
        profiling: &Qwen3TtsProfilingConfig,
        compiler: Qwen3TtsRequestCompiler,
    ) -> Result<Self, Qwen3TtsLoadError> {
        let inner = runtime::load_backend_runtime(package, backend, profiling, compiler)?;
        Ok(Self { inner })
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

    fn finish(self) -> Result<tts_core::PcmAudio, Self::Error> {
        self.inner.finish()
    }
}

#[derive(Debug)]
struct SessionImpl<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
    run: TalkerGenerator<B>,
    reference_codec_frames: Option<Vec<Vec<i64>>>,
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

    fn finish(self: Box<Self>) -> Result<tts_core::PcmAudio, Qwen3TtsInferenceError> {
        self.inner
            .finalize_audio(&self.run, self.reference_codec_frames.as_deref())
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
    let reference_codec_frames = seed.reference_codec_frames.clone();
    let run = inner.start_generator(seed, options)?;
    Ok(SessionImpl {
        inner,
        run,
        reference_codec_frames,
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

fn reference_codec_prefix_tensor<B: Backend>(
    reference_codec_frames: &[Vec<i64>],
    batch_size: usize,
    num_quantizers: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3, Int>, Qwen3TtsInferenceError> {
    if batch_size != 1 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("reference codec prefix only supports batch_size=1, got {batch_size}"),
        });
    }

    let flat = flatten_reference_codec_frames(reference_codec_frames, num_quantizers)?;
    Ok(Tensor::<B, 3, Int>::from_data(
        TensorData::new(
            flat,
            [batch_size, num_quantizers, reference_codec_frames.len()],
        ),
        device,
    ))
}

fn flatten_reference_codec_frames(
    reference_codec_frames: &[Vec<i64>],
    num_quantizers: usize,
) -> Result<Vec<i32>, Qwen3TtsInferenceError> {
    let mut flat = Vec::with_capacity(num_quantizers * reference_codec_frames.len());
    for group_idx in 0..num_quantizers {
        for (frame_idx, frame) in reference_codec_frames.iter().enumerate() {
            let value = frame
                .get(group_idx)
                .copied()
                .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "reference codec frame {frame_idx} has {} quantizers, expected at least {num_quantizers}",
                        frame.len()
                    ),
                })?;
            flat.push(i32::try_from(value).map_err(|_| Qwen3TtsInferenceError::InvalidInput {
                message: format!(
                    "reference codec token {value} at frame {frame_idx}, quantizer {group_idx} does not fit i32"
                ),
            })?);
        }
    }
    Ok(flat)
}

#[cfg(test)]
mod tests {
    use super::flatten_reference_codec_frames;

    #[test]
    fn flatten_reference_codec_frames_uses_quantizer_major_layout() {
        let frames = vec![vec![10, 20, 30], vec![11, 21, 31]];
        let flat = flatten_reference_codec_frames(&frames, 3).expect("frames should flatten");
        assert_eq!(flat, vec![10, 11, 20, 21, 30, 31]);
    }

    #[test]
    fn flatten_reference_codec_frames_rejects_short_frame() {
        let frames = vec![vec![10, 20], vec![11, 21, 31]];
        let error =
            flatten_reference_codec_frames(&frames, 3).expect_err("short frame should be rejected");
        let message = error.to_string();
        assert!(message.contains("reference codec frame 0"));
    }
}
