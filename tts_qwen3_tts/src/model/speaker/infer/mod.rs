use burn::tensor::backend::Backend;

pub(crate) mod feature;

use super::weights::LoadedQwen3TtsSpeakerEncoder;
use crate::Qwen3TtsInferenceError;
use crate::model::nn::tensor::read_float_tensor_vec;

impl<B> LoadedQwen3TtsSpeakerEncoder<B>
where
    B: Backend,
    B::Device: Clone,
{
    // TODO 不太合理频繁的转换和复制，应该直接输出tensor或者提供一个更高效的接口
    pub(crate) fn encode(&self, samples: &[f32]) -> Result<Vec<f32>, Qwen3TtsInferenceError> {
        let mel = self
            .mel_extractor
            .compute_for_speaker_encoder::<B>(samples, &self.device);
        let embed = self
            .encoder
            .forward(mel.unsqueeze_dim::<3>(0).cast(self.encoder.dtype()));
        read_float_tensor_vec(
            embed.reshape([self.encoder.enc_dim]),
            "failed to read speaker embedding",
        )
    }
}
