use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

#[derive(burn::module::Module, Debug)]
pub struct Qwen3RotaryEncoding<B: Backend> {
    inv_freq: Tensor<B, 1>,
    interleave_source: Tensor<B, 1, Int>,
    mrope_section: Vec<usize>,
}

#[derive(burn::module::Module, Debug)]
pub struct Qwen3StandardRotaryEncoding<B: Backend> {
    inv_freq: Tensor<B, 1>,
}

impl<B: Backend> Qwen3StandardRotaryEncoding<B> {
    pub fn new(head_dim: usize, rope_theta: f64, device: &B::Device) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..half_dim)
            .map(|index| 1.0f32 / (rope_theta as f32).powf(index as f32 / half_dim as f32))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
        Self { inv_freq }
    }

    pub fn get_cos_sin(
        &self,
        batch_size: usize,
        seq_len: usize,
        start: usize,
        dtype: DType,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half_dim = self.inv_freq.dims()[0];
        let positions = Tensor::<B, 1, burn::tensor::Int>::arange(
            start as i64..(start + seq_len) as i64,
            device,
        )
        .float()
        .reshape([1, seq_len, 1]);
        let freqs = positions * self.inv_freq.clone().reshape([1, 1, half_dim]);
        let cos_half = freqs.clone().cos();
        let sin_half = freqs.sin();
        let cos = Tensor::cat(vec![cos_half.clone(), cos_half], 2)
            .cast(dtype)
            .unsqueeze_dim::<4>(1)
            .repeat_dim(0, batch_size);
        let sin = Tensor::cat(vec![sin_half.clone(), sin_half], 2)
            .cast(dtype)
            .unsqueeze_dim::<4>(1)
            .repeat_dim(0, batch_size);
        (cos, sin)
    }
}

impl<B: Backend> Qwen3RotaryEncoding<B> {
    pub fn new(
        head_dim: usize,
        rope_theta: f64,
        mrope_section: Vec<usize>,
        device: &B::Device,
    ) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..half_dim)
            .map(|index| 1.0f32 / (rope_theta as f32).powf(index as f32 / half_dim as f32))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
        let interleave_source = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                interleaved_mrope_source_indices(&mrope_section, half_dim),
                [half_dim],
            ),
            device,
        );
        Self {
            inv_freq,
            interleave_source,
            mrope_section,
        }
    }

    /// Calculate the cos and sin tensors for multimodal rotary encoding.
    /// Returns (cos, sin) each with shape [batch_size, 1, seq_len, head_dim]
    ///
    /// Matches PyTorch apply_interleaved_rope (mrope_interleaved=True):
    ///   x_t = x[0].clone()
    ///   for beg, end in [(1,60), (2,60)]:
    ///       x_t[..., beg:end:3] = x[beg, ..., beg:end:3]
    ///
    pub fn get_cos_sin(
        &self,
        position_ids: Tensor<B, 3, burn::tensor::Int>,
        dtype: DType,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [modalities, batch_size, seq_len] = position_ids.dims();
        let half_dim = self.inv_freq.dims()[0];
        assert_eq!(
            modalities,
            self.mrope_section.len(),
            "position_ids modalities must match configured mrope sections"
        );

        // Compute cos/sin per modality in one tensor pass: [M, B, S, half_dim].
        let freqs = position_ids.float().unsqueeze_dim::<4>(3)
            * self.inv_freq.clone().reshape([1, 1, 1, half_dim]);
        let all_cos = freqs
            .clone()
            .cos()
            .swap_dims(0, 1)
            .swap_dims(1, 2)
            .swap_dims(2, 3);
        let all_sin = freqs.sin().swap_dims(0, 1).swap_dims(1, 2).swap_dims(2, 3);

        // Select the configured source modality for each half-dim position: [B, S, half_dim, 1].
        let source = self
            .interleave_source
            .clone()
            .reshape([1, 1, half_dim, 1])
            .repeat_dim(0, batch_size)
            .repeat_dim(1, seq_len);
        let cos_half = all_cos
            .gather(3, source.clone())
            .reshape([batch_size, seq_len, half_dim]);
        let sin_half = all_sin
            .gather(3, source)
            .reshape([batch_size, seq_len, half_dim]);

        // Duplicate half → full head_dim and unsqueeze for broadcast.
        (
            Tensor::cat(vec![cos_half.clone(), cos_half], 2)
                .cast(dtype)
                .unsqueeze_dim::<4>(1),
            Tensor::cat(vec![sin_half.clone(), sin_half], 2)
                .cast(dtype)
                .unsqueeze_dim::<4>(1),
        )
    }
}

fn interleaved_mrope_source_indices(mrope_section: &[usize], half_dim: usize) -> Vec<i64> {
    let modalities = mrope_section.len();
    (0..half_dim)
        .map(|pos| {
            let modality = pos % modalities;
            if modality > 0 && pos / modalities < mrope_section[modality] {
                modality as i64
            } else {
                0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Qwen3RotaryEncoding, interleaved_mrope_source_indices};
    use burn::backend::Flex;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    #[test]
    fn interleaved_mrope_source_indices_matches_expected_pattern() {
        let indices = interleaved_mrope_source_indices(&[2, 1, 1], 6);
        assert_eq!(indices, vec![0, 1, 2, 0, 0, 0]);
    }

    #[test]
    fn get_cos_sin_uses_configured_modality_per_half_dim() {
        let device = Default::default();
        let rope = Qwen3RotaryEncoding::<Flex>::new(6, 1.0, vec![1, 1, 1], &device);
        let position_ids = Tensor::<Flex, 3, Int>::from_data(
            TensorData::new(vec![0_i64, 1_i64, 2_i64], [3, 1, 1]),
            &device,
        );

        let (cos, sin) = rope.get_cos_sin(position_ids, DType::F32);
        let cos_values = cos
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cos should be readable");
        let sin_values = sin
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("sin should be readable");

        let expected_cos = vec![
            1.0_f32,
            1.0_f32.cos(),
            2.0_f32.cos(),
            1.0_f32,
            1.0_f32.cos(),
            2.0_f32.cos(),
        ];
        let expected_sin = vec![
            0.0_f32,
            1.0_f32.sin(),
            2.0_f32.sin(),
            0.0_f32,
            1.0_f32.sin(),
            2.0_f32.sin(),
        ];

        assert_eq!(cos_values.len(), expected_cos.len());
        assert_eq!(sin_values.len(), expected_sin.len());
        for (actual, expected) in cos_values.iter().zip(expected_cos.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in sin_values.iter().zip(expected_sin.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
