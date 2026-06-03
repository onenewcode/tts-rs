use burn::module::Module;
use burn::nn::{Linear, RmsNorm};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Tensor};

use super::kv::KeyValueCache;

pub enum AttentionPosition<'a, B: Backend> {
    Standard {
        cos: &'a Tensor<B, 4>,
        sin: &'a Tensor<B, 4>,
    },
    Mrope {
        cos: &'a Tensor<B, 4>,
        sin: &'a Tensor<B, 4>,
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
        position: AttentionPosition<'_, B>,
        mask: Option<&Tensor<B, 4, Bool>>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let x = x.dequantize();
        let [batch_size, seq_len, _] = x.dims();
        let model_dtype = x.dtype();
        let use_fp32_attention = model_dtype.size() < DType::F32.size();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = self
            .q_norm
            .forward(q.reshape([batch_size, seq_len, num_heads, head_dim]))
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(k.reshape([batch_size, seq_len, num_kv_heads, head_dim]))
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let (q, k, v) = if use_fp32_attention {
            (q.cast(DType::F32), k.cast(DType::F32), v.cast(DType::F32))
        } else {
            (q, k, v)
        };
        let (q, k) = match position {
            AttentionPosition::Standard { cos, sin } | AttentionPosition::Mrope { cos, sin } => {
                let rotary_dtype = q.dtype();
                let cos = cos.clone().dequantize().cast(rotary_dtype);
                let sin = sin.clone().dequantize().cast(rotary_dtype);
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
            use_fp32_attention,
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
        use_fp32_attention: bool,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<&Tensor<B, 4, Bool>>,
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

        #[allow(clippy::cast_precision_loss)]
        let scaling = (head_dim as f32).sqrt().recip();
        let attn_scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scaling);
        let attn_scores = if let Some(mask) = mask {
            attn_scores.mask_fill(mask.clone(), f32::NEG_INFINITY)
        } else {
            attn_scores
        };
        let attn_weights = softmax(attn_scores, 3);
        let attn_output = attn_weights.matmul(v);
        let attn_output = if use_fp32_attention {
            attn_output.cast(model_dtype)
        } else {
            attn_output
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
