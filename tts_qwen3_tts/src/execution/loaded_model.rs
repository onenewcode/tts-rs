use std::sync::Arc;

use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::compiler::Qwen3TtsRequestCompiler;
use super::compiler::session_seed::{SessionSeed, materialize_session_seed};
use super::run::LoadedModel;
use super::session::{ModelSession, SessionStep};
use crate::loading::runtime::{LoadedRuntime, RuntimeBackend};
use crate::model::codec::infer::Waveform;
use crate::model::codec::weights::LoadedQwen3TtsAudioCodec;
use crate::model::speaker::LoadedQwen3TtsSpeakerEncoder;
use crate::model::talker::infer::TalkerGenerator;
use crate::model::talker::infer::TalkerGeneratorStart;
use crate::model::talker::infer::sampling::SamplingConfig as RuntimeSamplingConfig;
use crate::model::talker::weights::LoadedQwen3TtsTalker;
use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsInferenceError, Qwen3TtsRunOptions,
    Qwen3TtsVoiceClonePrompt, QwenRequest,
};

#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsLoadedModel {
    runtime: Arc<LoadedRuntime<RuntimeBackend>>,
}

impl Qwen3TtsLoadedModel {
    pub(crate) fn new(runtime: LoadedRuntime<RuntimeBackend>) -> Self {
        Self {
            runtime: Arc::new(runtime),
        }
    }

    pub(crate) fn create_voice_clone_prompt(
        &self,
        reference: &BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError> {
        match self.runtime.as_ref() {
            LoadedRuntime::BaseVoiceClone {
                core,
                speaker_encoder,
            } => crate::execution::conditioning::create_voice_clone_prompt(
                &core.decoder,
                speaker_encoder.as_ref(),
                &core.device,
                reference,
            ),
            LoadedRuntime::BaseSynthesis(_) | LoadedRuntime::CustomVoice(_) => {
                Err(Qwen3TtsInferenceError::RuntimeLoad {
                    message: "loaded runtime does not include speaker encoder support".to_string(),
                })
            }
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
            inner: start_session_impl(&self.runtime, request, options)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Qwen3TtsSession {
    inner: SessionImpl,
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
struct SessionImpl {
    runtime: Arc<LoadedRuntime<RuntimeBackend>>,
    run: TalkerGenerator<RuntimeBackend>,
    reference_codec_prefix: Option<Tensor<RuntimeBackend, 3, Int>>,
    reference_codec_frame_count: usize,
}

impl SessionImpl {
    fn step(&mut self) -> Result<SessionStep, Qwen3TtsInferenceError> {
        let step_result = self
            .run
            .step(talker_runtime(self.runtime.as_ref())?.talker)?;
        match step_result {
            Some(step) if step.finished => Ok(SessionStep::Finished),
            Some(_) => Ok(SessionStep::Advanced),
            None => Ok(SessionStep::Finished),
        }
    }

    fn finish(self) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        let runtime = talker_runtime(self.runtime.as_ref())?;
        let generated = self.run.finalize()?;
        let waveform = if let Some(reference_codec_prefix) = self.reference_codec_prefix {
            let [batch_size, num_quantizers, time_steps] = generated.codec_token_ids.dims();
            let [prefix_batch, prefix_quantizers, prefix_steps] = reference_codec_prefix.dims();
            if prefix_batch != batch_size || prefix_quantizers != num_quantizers {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "reference codec prefix shape mismatch: expected [{batch_size}, {num_quantizers}, T], got [{prefix_batch}, {prefix_quantizers}, {prefix_steps}]"
                    ),
                });
            }
            let combined_steps = time_steps + self.reference_codec_frame_count;
            let codec_ids = Tensor::cat(vec![reference_codec_prefix, generated.codec_token_ids], 2);
            let mut waveform = runtime.decoder.decode_waveform(codec_ids)?;
            let total_samples = waveform.dims()[2];
            let cut_samples =
                self.reference_codec_frame_count * total_samples / combined_steps.max(1);
            waveform = waveform.slice([0..1, 0..1, cut_samples.min(total_samples)..total_samples]);
            waveform
        } else {
            runtime.decoder.decode_waveform(generated.codec_token_ids)?
        };
        let waveform = Waveform::from_tensor(
            u32::try_from(runtime.decoder.config.output_sample_rate).map_err(|_| {
                Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "decoder output sample rate {} exceeds the supported u32 audio range",
                        runtime.decoder.config.output_sample_rate
                    ),
                }
            })?,
            waveform,
        )?;
        Ok(tts_infer::PcmAudio {
            pcm_i16: waveform.to_pcm(),
            sample_rate: waveform.sample_rate(),
            channels: 1,
        })
    }
}

fn start_session_impl(
    runtime: &Arc<LoadedRuntime<RuntimeBackend>>,
    request: QwenRequest,
    options: Qwen3TtsRunOptions,
) -> Result<SessionImpl, Qwen3TtsInferenceError> {
    let runtime = Arc::clone(runtime);
    let runtime_view = talker_runtime(runtime.as_ref())?;
    let condition = runtime_view.compiler.compile_request(&request)?;
    let seed = materialize_session_seed(
        &condition,
        &runtime_view.talker.config,
        runtime_view.talker,
        runtime_view.decoder,
        runtime_view.speaker_encoder,
        runtime_view.tensor_dtype,
        runtime_view.device,
    )?;
    let SessionSeed {
        inputs_embeds,
        position_ids,
        attention_mask,
        trailing_text_hidden,
        tts_pad_embed,
        reference_codec_prefix,
        reference_codec_frame_count,
        max_new_tokens,
        codec_eos_token_id,
        sampling: seed_sampling,
        suppress_token_ids,
    } = seed;
    let run = TalkerGenerator::start(
        &runtime_view.talker.config,
        runtime_view.talker,
        TalkerGeneratorStart {
            inputs_embeds,
            position_ids,
            attention_mask,
            trailing_text_hidden,
            tts_pad_embed,
            sampling: resolve_sampling(options.sampling.as_ref(), &seed_sampling),
            max_new_tokens: options.max_new_tokens.unwrap_or(max_new_tokens),
            eos_token_id: Some(codec_eos_token_id),
            suppress_token_ids,
        },
    )?;
    Ok(SessionImpl {
        runtime,
        run,
        reference_codec_prefix,
        reference_codec_frame_count,
    })
}

struct TalkerRuntimeView<'a, B: Backend> {
    device: &'a B::Device,
    tensor_dtype: DType,
    compiler: &'a Qwen3TtsRequestCompiler,
    talker: &'a LoadedQwen3TtsTalker<B>,
    decoder: &'a LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: Option<&'a LoadedQwen3TtsSpeakerEncoder<B>>,
}

fn talker_runtime(
    runtime: &LoadedRuntime<RuntimeBackend>,
) -> Result<TalkerRuntimeView<'_, RuntimeBackend>, Qwen3TtsInferenceError> {
    match runtime {
        LoadedRuntime::BaseSynthesis(core) | LoadedRuntime::CustomVoice(core) => {
            Ok(TalkerRuntimeView {
                device: &core.device,
                tensor_dtype: core.tensor_dtype,
                compiler: &core.compiler,
                talker: &core.talker,
                decoder: &core.decoder,
                speaker_encoder: None,
            })
        }
        LoadedRuntime::BaseVoiceClone {
            core,
            speaker_encoder,
        } => Ok(TalkerRuntimeView {
            device: &core.device,
            tensor_dtype: core.tensor_dtype,
            compiler: &core.compiler,
            talker: &core.talker,
            decoder: &core.decoder,
            speaker_encoder: Some(speaker_encoder.as_ref()),
        }),
    }
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
