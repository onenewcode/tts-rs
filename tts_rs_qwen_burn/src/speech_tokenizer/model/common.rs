use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, ConvTranspose1d};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct TokenizerCausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct TokenizerCausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

#[derive(Module, Debug)]
pub struct TokenizerSnakeBeta<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct TokenizerLayerScale<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug, Default, Clone)]
pub struct Qwen3TtsSpeechTokenizerEmptyModule;

// ---- Forward implementations ----

impl<B: Backend> TokenizerCausalConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}

impl<B: Backend> TokenizerCausalTransConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(x)
    }
}

impl<B: Backend> TokenizerSnakeBeta<B> {
    /// SnakeBeta activation: `x + sin^2(alpha * x) / (beta + eps)`
    /// Supports both `[B, C, T]` (CNN) and `[B, S, H]` (Transformer) formats.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = 1e-8;
        let n_param = self.alpha.dims()[0];
        let [d0, d1, d2] = x.dims();
        // Determine which axis carries the feature/channel dimension
        let (alpha, beta) = if d1 == n_param {
            // [B, C, T] — CNN format
            (self.alpha.val().reshape([1, n_param, 1]),
             self.beta.val().reshape([1, n_param, 1]))
        } else {
            // [B, S, H] — Transformer format
            (self.alpha.val().reshape([1, 1, n_param]),
             self.beta.val().reshape([1, 1, n_param]))
        };
        let sin_sq = (x.clone().mul(alpha)).sin().powf_scalar(2.0);
        x + sin_sq.div(beta.add_scalar(eps))
    }
}

impl<B: Backend> TokenizerLayerScale<B> {
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
