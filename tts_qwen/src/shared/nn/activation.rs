// SnakeBeta activation and LayerScale — shared by talker and audio codec.
use burn::module::{Module, Param};
use burn::tensor::{DType, Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct AudioCodecSnakeBeta<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct AudioCodecLayerScale<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug, Default, Clone)]
pub struct Qwen3TtsAudioCodecEmptyModule;

impl<B: Backend> AudioCodecSnakeBeta<B> {
    /// SnakeBeta activation: `x + sin^2(exp(alpha) * x) / (exp(beta) + eps)`.
    /// Supports both `[B, C, T]` (CNN) and `[B, S, H]` (Transformer) formats.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = x.dtype();
        let x_f32 = x.cast(DType::F32);
        let eps = 1e-9;
        let n_param = self.alpha.dims()[0];
        let [_, d1, _] = x_f32.dims();
        let (alpha, beta) = if d1 == n_param {
            (
                self.alpha
                    .val()
                    .cast(DType::F32)
                    .exp()
                    .reshape([1, n_param, 1]),
                self.beta
                    .val()
                    .cast(DType::F32)
                    .exp()
                    .reshape([1, n_param, 1]),
            )
        } else {
            (
                self.alpha
                    .val()
                    .cast(DType::F32)
                    .exp()
                    .reshape([1, 1, n_param]),
                self.beta
                    .val()
                    .cast(DType::F32)
                    .exp()
                    .reshape([1, 1, n_param]),
            )
        };
        let sin_sq = (x_f32.clone().mul(alpha)).sin().powf_scalar(2.0);
        (x_f32 + sin_sq.div(beta.add_scalar(eps))).cast(dtype)
    }
}

impl<B: Backend> AudioCodecLayerScale<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let n_param = self.scale.dims()[0];
        let [_, d1, _] = x.dims();
        let scale = if d1 == n_param {
            self.scale.val().reshape([1, n_param, 1])
        } else {
            self.scale.val().reshape([1, 1, n_param])
        };
        x.mul(scale)
    }
}

impl<B: Backend> AudioCodecSnakeBeta<B> {
    pub(crate) fn new(channels: usize, device: &B::Device) -> Self {
        use burn::module::Initializer;
        Self {
            alpha: Initializer::Zeros.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }
}

impl<B: Backend> AudioCodecLayerScale<B> {
    pub(crate) fn new(channels: usize, initial_scale: f64, device: &B::Device) -> Self {
        use burn::module::Initializer;
        Self {
            scale: Initializer::Constant {
                value: initial_scale,
            }
            .init([channels], device),
        }
    }
}
