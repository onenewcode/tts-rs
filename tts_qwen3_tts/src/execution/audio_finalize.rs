use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::Qwen3TtsInferenceError;
pub(crate) fn reference_codec_prefix_tensor<B: Backend>(
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
    Ok(
        Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape([
            batch_size,
            num_quantizers,
            reference_codec_frames.len(),
        ]),
    )
}

fn flatten_reference_codec_frames(
    reference_codec_frames: &[Vec<i64>],
    num_quantizers: usize,
) -> Result<Vec<i64>, Qwen3TtsInferenceError> {
    let mut flat = Vec::with_capacity(num_quantizers * reference_codec_frames.len());
    for group_idx in 0..num_quantizers {
        for (frame_idx, frame) in reference_codec_frames.iter().enumerate() {
            if frame.len() < num_quantizers {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: format!(
                        "reference codec frame {frame_idx} has {} quantizers, expected at least {num_quantizers}",
                        frame.len()
                    ),
                });
            }
            flat.push(frame[group_idx]);
        }
    }
    Ok(flat)
}
