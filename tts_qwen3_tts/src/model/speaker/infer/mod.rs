use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub(crate) mod feature;

use super::weights::LoadedQwen3TtsSpeakerEncoder;

impl<B> LoadedQwen3TtsSpeakerEncoder<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn encode_embedding(&self, samples: &[f32]) -> Tensor<B, 1> {
        let mel = self
            .mel_extractor
            .compute_for_speaker_encoder::<B>(samples, &self.device)
            .unsqueeze_dim::<3>(0);
        let mel = if mel.dtype() == self.encoder.dtype() {
            mel
        } else {
            mel.cast(self.encoder.dtype())
        };
        self.encoder.forward(mel).reshape([self.encoder.enc_dim])
    }
}
