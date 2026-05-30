use burn::tensor::backend::Backend;

use super::weights::LoadedQwen3TtsSpeakerEncoder;
use crate::Qwen3TtsInferenceError;

impl<B> LoadedQwen3TtsSpeakerEncoder<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn encode(&self, samples: &[f32]) -> Result<Vec<f32>, Qwen3TtsInferenceError> {
        let mel = self
            .mel_extractor
            .compute_for_speaker_encoder::<B>(samples, &self.device);
        let embed = self
            .encoder
            .forward(mel.unsqueeze_dim::<3>(0).cast(self.encoder.dtype()));
        embed
            .reshape([self.encoder.enc_dim])
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|source| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read speaker embedding: {source}"),
            })
    }
}
