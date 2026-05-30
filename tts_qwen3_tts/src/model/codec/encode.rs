use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::Qwen3TtsInferenceError;

use super::weights::LoadedQwen3TtsAudioCodec;

pub fn encode_reference_codec_frames<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    device: &B::Device,
    samples: &[f32],
) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
    let waveform = Tensor::<B, 3>::from_data(
        TensorData::new(samples.to_vec(), [1, 1, samples.len()]),
        device,
    );
    loaded.model.encoder.encode_reference_frames(
        &loaded.config.encoder_config,
        loaded.config.encoder_valid_num_quantizers,
        waveform,
    )
}
