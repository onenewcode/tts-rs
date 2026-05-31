use burn::tensor::backend::Backend;

use crate::execution::reference_audio::load_reference_audio;
use crate::model::codec::loading::LoadedQwen3TtsAudioCodec;
use crate::model::codec::runtime::encode_reference_codec_frames;
use crate::model::speaker::LoadedQwen3TtsSpeakerEncoder;
use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsInferenceError, Qwen3TtsVoiceClonePrompt,
    Qwen3TtsVoiceClonePromptMode,
};

pub(crate) fn create_voice_clone_prompt<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: &LoadedQwen3TtsSpeakerEncoder<B>,
    device: &B::Device,
    reference: &BaseVoiceCloneReferenceAudio,
) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError>
where
    B::Device: Clone,
{
    let transcript = reference
        .transcript
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if !reference.x_vector_only && transcript.is_none() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "ref_text is required when x_vector_only is false".to_string(),
        });
    }

    let prepared_for_speaker =
        load_reference_audio(&reference.path, speaker_encoder.sample_rate())?;
    let speaker_embedding = speaker_encoder.encode(&prepared_for_speaker.samples)?;
    if speaker_embedding.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "reference audio {} produced no speaker embedding",
                reference.path.display()
            ),
        });
    }

    let ref_codec_token_ids = if reference.x_vector_only {
        None
    } else {
        Some(encode_reference_codec_frames(
            loaded,
            device,
            &load_reference_audio(&reference.path, loaded.config.input_sample_rate as u32)?.samples,
        )?)
    };

    Ok(Qwen3TtsVoiceClonePrompt {
        speaker_embedding,
        ref_codec_token_ids,
        transcript: transcript.map(ToOwned::to_owned),
        mode: if reference.x_vector_only {
            Qwen3TtsVoiceClonePromptMode::XVectorOnly
        } else {
            Qwen3TtsVoiceClonePromptMode::Icl
        },
    })
}
