use burn::tensor::DType;
use burn::tensor::backend::Backend;

use crate::Qwen3TtsLoadError;
use crate::execution::compiler::Qwen3TtsRequestCompiler;
use crate::loading::ResolvedPackage;
use crate::model::codec::weights::{LoadedQwen3TtsAudioCodec, load_qwen3_tts_audio_codec};
use crate::model::speaker::weights::{
    LoadedQwen3TtsSpeakerEncoder, load_qwen3_tts_speaker_encoder,
};
use crate::model::talker::weights::{LoadedQwen3TtsTalker, load_qwen3_tts_talker_for_inference};

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

#[cfg(any(
    all(
        feature = "flex",
        any(
            feature = "wgpu",
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "webgpu"
        )
    ),
    all(
        feature = "wgpu",
        any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "webgpu"
        )
    ),
    all(
        feature = "cuda",
        any(
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "webgpu"
        )
    ),
    all(
        feature = "rocm",
        any(feature = "metal", feature = "vulkan", feature = "webgpu")
    ),
    all(feature = "metal", any(feature = "vulkan", feature = "webgpu")),
    all(feature = "vulkan", feature = "webgpu"),
))]
compile_error!("enable exactly one backend feature for tts_qwen3_tts");

#[cfg(feature = "flex")]
pub(crate) type RuntimeBackend = burn::backend::Flex;
#[cfg(feature = "wgpu")]
pub(crate) type RuntimeBackend = burn::backend::Wgpu;
#[cfg(feature = "cuda")]
pub(crate) type RuntimeBackend = burn::backend::Cuda;
#[cfg(feature = "rocm")]
pub(crate) type RuntimeBackend = burn::backend::Rocm;
#[cfg(feature = "metal")]
pub(crate) type RuntimeBackend = burn::backend::Metal;
#[cfg(feature = "vulkan")]
pub(crate) type RuntimeBackend = burn::backend::Vulkan;
#[cfg(feature = "webgpu")]
pub(crate) type RuntimeBackend = burn::backend::WebGpu;

#[derive(Debug)]
pub(crate) struct CoreSynthesisRuntime<B: Backend> {
    pub(crate) device: B::Device,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
    pub(crate) talker: LoadedQwen3TtsTalker<B>,
    pub(crate) decoder: LoadedQwen3TtsAudioCodec<B>,
}

#[derive(Debug)]
pub(crate) enum LoadedRuntime<B: Backend> {
    BaseSynthesis(CoreSynthesisRuntime<B>),
    BaseVoiceClone {
        core: CoreSynthesisRuntime<B>,
        speaker_encoder: Box<LoadedQwen3TtsSpeakerEncoder<B>>,
    },
    CustomVoice(CoreSynthesisRuntime<B>),
}

pub(crate) fn build_runtime<B: Backend>(
    resolved: &ResolvedPackage,
    talker_dtype: Option<DType>,
    codec_dtype: Option<DType>,
    device: &B::Device,
) -> Result<LoadedRuntime<B>, Qwen3TtsLoadError>
where
    B::Device: Clone,
{
    if resolved.compiler.profiles.custom_voice.is_some() {
        return Ok(LoadedRuntime::CustomVoice(load_core_runtime::<B>(
            resolved,
            talker_dtype,
            codec_dtype,
            device,
        )?));
    }

    let core = load_core_runtime::<B>(resolved, talker_dtype, codec_dtype, device)?;
    if resolved.has_speaker_encoder {
        Ok(LoadedRuntime::BaseVoiceClone {
            core,
            speaker_encoder: Box::new(load_qwen3_tts_speaker_encoder::<B>(
                &resolved.package.talker_config_path,
                &resolved.package.talker_weights_path,
                device,
                talker_dtype,
            )?),
        })
    } else {
        Ok(LoadedRuntime::BaseSynthesis(core))
    }
}

fn load_core_runtime<B: Backend>(
    resolved: &ResolvedPackage,
    talker_dtype: Option<DType>,
    codec_dtype: Option<DType>,
    device: &B::Device,
) -> Result<CoreSynthesisRuntime<B>, Qwen3TtsLoadError>
where
    B::Device: Clone,
{
    Ok(CoreSynthesisRuntime {
        device: device.clone(),
        compiler: resolved.compiler.clone(),
        talker: load_qwen3_tts_talker_for_inference::<B>(
            &resolved.package.talker_config_path,
            &resolved.package.talker_weights_path,
            device,
            talker_dtype,
        )?,
        decoder: load_qwen3_tts_audio_codec::<B>(
            &resolved.package.codec_config_path,
            &resolved.package.codec_weights_path,
            device,
            codec_dtype,
        )?,
    })
}
