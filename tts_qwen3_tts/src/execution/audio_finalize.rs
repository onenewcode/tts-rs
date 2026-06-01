use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::Qwen3TtsInferenceError;
// TODO 优化这部分代码
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
