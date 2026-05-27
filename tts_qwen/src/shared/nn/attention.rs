use burn::module::Module;
use burn::nn::{Linear, RmsNorm};
use burn::tensor::DType;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::mlp::native_linear_3d;
use super::rms_norm::qwen_rms_norm;
use crate::talker::KeyValueCache;

pub struct AttentionOutput<B: Backend> {
    pub output: Tensor<B, 3>,
    pub attn_weights: Tensor<B, 4>,
    pub q_proj: Tensor<B, 3>,
    pub k_proj: Tensor<B, 3>,
    pub v_proj: Tensor<B, 3>,
    pub q_norm: Tensor<B, 3>,
    pub k_norm: Tensor<B, 3>,
    pub q_rot: Tensor<B, 3>,
    pub k_rot: Tensor<B, 3>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionValueMode {
    CastSoftmaxToModelDTypeBeforeValueMatmul,
}

pub enum AttentionPosition<B: Backend> {
    Standard {
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    },
    Mrope {
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    },
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3TtsAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
        value_mode: AttentionValueMode,
    ) -> Tensor<B, 3> {
        self.forward_debug(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            position,
            mask,
            cache,
            value_mode,
        )
        .output
    }

    pub fn forward_debug(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
        value_mode: AttentionValueMode,
    ) -> AttentionOutput<B> {
        self.forward_debug_with_value_mode(
            x,
            num_heads,
            num_kv_heads,
            head_dim,
            position,
            mask,
            cache,
            value_mode,
        )
    }

    fn forward_debug_with_value_mode(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
        value_mode: AttentionValueMode,
    ) -> AttentionOutput<B> {
        let [batch_size, seq_len, _] = x.dims();

        let q = native_linear_3d(&self.q_proj, x.clone());
        let k = native_linear_3d(&self.k_proj, x.clone());
        let v = native_linear_3d(&self.v_proj, x);
        let q_proj_out = q.clone();
        let k_proj_out = k.clone();
        let v_proj_out = v.clone();

        // Apply norm PER HEAD (last dimension after reshape)
        let q = qwen_rms_norm(
            &self.q_norm,
            q.reshape([batch_size, seq_len, num_heads, head_dim]),
        )
        .swap_dims(1, 2)
        .clone();
        let k = qwen_rms_norm(
            &self.k_norm,
            k.reshape([batch_size, seq_len, num_kv_heads, head_dim]),
        )
        .swap_dims(1, 2)
        .clone();
        let v = v
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2)
            .clone();
        let q_norm_out =
            q.clone()
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);
        let k_norm_out =
            k.clone()
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_kv_heads * head_dim]);

        let (q, k) = match position {
            AttentionPosition::Standard { cos, sin } => {
                let q = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
                let k = (k.clone() * cos) + (rotate_half(k) * sin);
                (q, k)
            }
            AttentionPosition::Mrope { cos, sin } => {
                let q = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
                let k = (k.clone() * cos) + (rotate_half(k) * sin);
                (q, k)
            }
        };
        let q_rot_out =
            q.clone()
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);
        let k_rot_out =
            k.clone()
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_kv_heads * head_dim]);

        let (k, v) = cache.forward(k, v);

        let (output, attn_weights) = self.execute_attention(
            batch_size,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            q,
            k,
            v,
            mask,
            value_mode,
        );

        AttentionOutput {
            output,
            attn_weights,
            q_proj: q_proj_out,
            k_proj: k_proj_out,
            v_proj: v_proj_out,
            q_norm: q_norm_out,
            k_norm: k_norm_out,
            q_rot: q_rot_out,
            k_rot: k_rot_out,
        }
    }

    fn execute_attention(
        &self,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
        _value_mode: AttentionValueMode,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let n_rep = num_heads / num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        let dtype = q.dtype();
        let scaling = (head_dim as f32).sqrt().recip();
        let attn_scores = q
            .clone()
            .cast(DType::F32)
            .matmul(k.clone().cast(DType::F32).swap_dims(2, 3).clone())
            .mul_scalar(scaling);
        let attn_scores = if let Some(mask) = mask {
            attn_scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attn_scores
        };
        let attn_weights_f32 = softmax(attn_scores.cast(DType::F32), 3);
        let attn_weights = attn_weights_f32.clone().cast(dtype);
        let attn_output = attn_weights
            .clone()
            .cast(DType::F32)
            .matmul(v.cast(DType::F32))
            .cast(dtype);

        // clone after swap_dims to ensure contiguous layout (matching PyTorch's .contiguous())
        let attn_output = attn_output.swap_dims(1, 2).clone();
        let attn_output = attn_output.reshape([batch_size, seq_len, num_heads * head_dim]);
        (native_linear_3d(&self.o_proj, attn_output), attn_weights)
    }
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
        batch_size,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    ])
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, heads, seq_len, head_dim] = x.dims();
    let half_dim = head_dim / 2;
    let x1 = x
        .clone()
        .slice([0..batch_size, 0..heads, 0..seq_len, 0..half_dim]);
    let x2 = x.slice([0..batch_size, 0..heads, 0..seq_len, half_dim..head_dim]);
    Tensor::cat(vec![-x2, x1], 3)
}
