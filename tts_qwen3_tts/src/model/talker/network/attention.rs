use burn::module::Module;
use burn::nn::{Linear, RmsNorm};
use burn::tensor::DType;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::kv::KeyValueCache;
use crate::model::nn::attention::repeat_kv_heads;

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
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position: AttentionPosition<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = self
            .q_norm
            .forward(q.reshape([batch_size, seq_len, num_heads, head_dim]))
            .swap_dims(1, 2)
            .clone();
        let k = self
            .k_norm
            .forward(k.reshape([batch_size, seq_len, num_kv_heads, head_dim]))
            .swap_dims(1, 2)
            .clone();
        let v = v
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2)
            .clone();

        let (q, k) = match position {
            AttentionPosition::Standard { cos, sin } | AttentionPosition::Mrope { cos, sin } => {
                let q = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
                let k = (k.clone() * cos) + (rotate_half(k) * sin);
                (q, k)
            }
        };

        let (k, v) = cache.forward(k, v);

        self.execute_attention(
            batch_size,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            q,
            k,
            v,
            mask,
        )
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> Tensor<B, 3> {
        let n_rep = num_heads / num_kv_heads;
        let k = repeat_kv_heads(k, n_rep);
        let v = repeat_kv_heads(v, n_rep);

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

        let attn_output = attn_output.swap_dims(1, 2).clone();
        let attn_output = attn_output.reshape([batch_size, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attn_output)
    }
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
