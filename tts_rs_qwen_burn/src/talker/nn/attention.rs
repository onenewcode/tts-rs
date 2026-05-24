use burn::module::Module;
use burn::nn::{Linear, RmsNorm, RotaryEncoding};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, s};

use super::super::cache::KeyValueCache;

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
    /// Forward pass for Qwen3TtsAttention with standard RoPE (for CodePredictor)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: &RotaryEncoding<B>,
        mask: Tensor<B, 4, Bool>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Apply norm PER HEAD (last dimension after reshape)
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

        // Apply official RoPE
        let offset = cache.len();
        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

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

    /// Forward pass for Qwen3TtsAttention with pre-calculated multimodal RoPE tensors (for Talker)
    pub fn forward_mrope(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        mask: Tensor<B, 4, Bool>,
        cache: &mut KeyValueCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Apply norm PER HEAD (last dimension after reshape)
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

        // Apply mRoPE rotation
        let q = (q.clone() * cos.clone()) + (rotate_half(q) * sin.clone());
        let k = (k.clone() * cos) + (rotate_half(k) * sin);

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
        mask: Tensor<B, 4, Bool>,
    ) -> Tensor<B, 3> {
        let n_rep = num_heads / num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        let scale = (head_dim as f64).powf(-0.5);
        let attn_weights = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let attn_weights = attn_weights.mask_fill(mask, f32::NEG_INFINITY);
        let attn_weights = softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let attn_output =
            attn_output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attn_output)
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
