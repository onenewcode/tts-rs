use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

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
    pub fn new(head_dim: usize, rope_theta: f32, device: &B::Device) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..half_dim)
            .map(|index| 1.0f32 / rope_theta.powf(index as f32 / half_dim as f32))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
        Self { inv_freq }
    }

    pub fn get_cos_sin(
        &self,
        batch_size: usize,
        seq_len: usize,
        start: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half_dim = self.inv_freq.dims()[0];
        let start = i64::try_from(start).expect("rope start should fit into i64");
        let end = start
            .checked_add(i64::try_from(seq_len).expect("rope sequence length should fit into i64"))
            .expect("rope position range should fit into i64");
        let positions = Tensor::<B, 1, burn::tensor::Int>::arange(start..end, device)
            .float()
            .reshape([1, seq_len, 1]);
        let freqs = positions * self.inv_freq.clone().reshape([1, 1, half_dim]);
        let cos_half = freqs.clone().cos();
        let sin_half = freqs.sin();
        let cos = cos_half
            .repeat_dim(2, 2)
            .unsqueeze_dim::<4>(1)
            .repeat_dim(0, batch_size);
        let sin = sin_half
            .repeat_dim(2, 2)
            .unsqueeze_dim::<4>(1)
            .repeat_dim(0, batch_size);
        (cos, sin)
    }
}

impl<B: Backend> Qwen3RotaryEncoding<B> {
    pub fn new(
        head_dim: usize,
        rope_theta: f32,
        mrope_section: Vec<usize>,
        device: &B::Device,
    ) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..half_dim)
            .map(|index| 1.0f32 / rope_theta.powf(index as f32 / half_dim as f32))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
        let interleave_source = Tensor::<B, 1, Int>::from_ints(
            interleaved_mrope_source_indices(&mrope_section, half_dim).as_slice(),
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
        let cos = cos_half.repeat_dim(2, 2).unsqueeze_dim::<4>(1);
        let sin = sin_half.repeat_dim(2, 2).unsqueeze_dim::<4>(1);
        (cos, sin)
    }
}

fn interleaved_mrope_source_indices(mrope_section: &[usize], half_dim: usize) -> Vec<i64> {
    let modalities = mrope_section.len();
    (0..half_dim)
        .map(|pos| {
            let modality = pos % modalities;
            if modality > 0 && pos / modalities < mrope_section[modality] {
                i64::try_from(modality).expect("modality index should fit into i64")
            } else {
                0
            }
        })
        .collect()
}
