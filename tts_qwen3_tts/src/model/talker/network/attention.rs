use burn::module::Module;
use burn::nn::{Linear, RmsNorm};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Tensor};

use super::kv::KeyValueCache;

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
        let model_dtype = x.dtype();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = self
            .q_norm
            .forward(q.reshape([batch_size, seq_len, num_heads, head_dim]))
            .swap_dims(1, 2)
            .cast(DType::F32);
        let k = self
            .k_norm
            .forward(k.reshape([batch_size, seq_len, num_kv_heads, head_dim]))
            .swap_dims(1, 2)
            .cast(DType::F32);
        let v = v
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2)
            .cast(DType::F32);
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
            model_dtype,
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
        model_dtype: DType,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let n_rep = num_heads / num_kv_heads;
        let key_len = k.dims()[2];
        let k = k.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
            batch_size,
            num_kv_heads * n_rep,
            key_len,
            head_dim,
        ]);
        let value_len = v.dims()[2];
        let v = v.unsqueeze_dim::<5>(2).repeat_dim(2, n_rep).reshape([
            batch_size,
            num_kv_heads * n_rep,
            value_len,
            head_dim,
        ]);

        let scaling = (head_dim as f32).sqrt().recip();
        let attn_scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scaling);
        let attn_scores = if let Some(mask) = mask {
            attn_scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            attn_scores
        };
        let attn_weights = softmax(attn_scores, 3);
        let attn_output = attn_weights.matmul(v);
        let attn_output = if model_dtype == DType::F32 {
            attn_output
        } else {
            attn_output.cast(model_dtype)
        };

        let attn_output = attn_output.swap_dims(1, 2);
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
