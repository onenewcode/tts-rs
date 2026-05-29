use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, s};

/// TODO 你的rope应该抽取一个公共的特征
#[derive(burn::module::Module, Debug)]
pub struct Qwen3RotaryEncoding<B: Backend> {
    inv_freq: Tensor<B, 1>,
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
        Self {
            inv_freq,
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
    /// We implement this without strided slices by iterating each of the 64 half_dim
    /// positions and selecting the correct source modality via unit-stride slices.
    pub fn get_cos_sin(
        &self,
        position_ids: Tensor<B, 3, burn::tensor::Int>,
        dtype: DType,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [modalities, batch_size, seq_len] = position_ids.dims();
        let half_dim = self.inv_freq.dims()[0];

        // 1. Compute cos/sin per modality: stack to [M, B, S, half_dim]
        let inv_freq = self.inv_freq.clone().reshape([1, 1, half_dim]);
        let mut modality_cos = Vec::with_capacity(modalities);
        let mut modality_sin = Vec::with_capacity(modalities);
        for modality in 0..modalities {
            let ids = position_ids
                .clone()
                .slice_dim(0, modality..modality + 1)
                .reshape([batch_size, seq_len, 1])
                .float();
            let freqs = ids * inv_freq.clone();
            modality_cos.push(freqs.clone().cos());
            modality_sin.push(freqs.sin());
        }
        let all_cos =
            Tensor::cat(modality_cos, 0).reshape([modalities, batch_size, seq_len, half_dim]);
        let all_sin =
            Tensor::cat(modality_sin, 0).reshape([modalities, batch_size, seq_len, half_dim]);

        // 2. Interleave per-position (matching PyTorch apply_interleaved_rope):
        //    x_t = x[0]; x_t[pos] = x[m][pos] where pos%M == m && pos/M < section_len[m]
        let mut cos_half = all_cos
            .clone()
            .slice_dim(0, 0..1)
            .reshape([batch_size, seq_len, half_dim])
            .clone();
        let mut sin_half = all_sin
            .clone()
            .slice_dim(0, 0..1)
            .reshape([batch_size, seq_len, half_dim])
            .clone();
        for pos in 0..half_dim {
            let m = pos % modalities;
            if m > 0 && pos / modalities < self.mrope_section[m] {
                let src_cos = all_cos
                    .clone()
                    .slice_dim(0, m..m + 1)
                    .reshape([batch_size, seq_len, half_dim])
                    .slice(s![.., .., pos..pos + 1]);
                let src_sin = all_sin
                    .clone()
                    .slice_dim(0, m..m + 1)
                    .reshape([batch_size, seq_len, half_dim])
                    .slice(s![.., .., pos..pos + 1]);
                cos_half = cos_half.slice_assign(s![.., .., pos..pos + 1], src_cos);
                sin_half = sin_half.slice_assign(s![.., .., pos..pos + 1], src_sin);
            }
        }

        // 3. Duplicate half → full head_dim and unsqueeze for broadcast
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
