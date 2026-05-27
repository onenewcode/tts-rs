use burn::tensor::{Tensor, backend::Backend};

/// Autoregressive cache for a single tensor (e.g., Key or Value).
#[derive(Debug)]
pub struct AutoregressiveCache<B: Backend> {
    /// Tensor cache with shape `[batch_size, num_heads, max_seq_len, head_dim]`
    cache: Option<Tensor<B, 4>>,
    pub max_seq_len: usize,
    cur_seq_len: usize,
    // Configuration for lazy initialization
    max_batch_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> AutoregressiveCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            cache: None,
            max_seq_len,
            cur_seq_len: 0,
            max_batch_size,
            num_heads,
            head_dim,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        tracing::debug!(
            previous_seq_len = self.cur_seq_len,
            max_seq_len = self.max_seq_len,
            "reset autoregressive cache"
        );
        self.cache = None;
        self.cur_seq_len = 0;
    }

    /// Update the cache and return the current valid slice.
    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();
        let original_tensor = tensor.clone();
        let old_seq_len = self.cur_seq_len;
        assert!(
            batch_size <= self.max_batch_size,
            "cache batch size exceeded"
        );
        assert_eq!(num_heads, self.num_heads, "cache head count mismatch");
        assert_eq!(head_dim, self.head_dim, "cache head dim mismatch");
        let mut new_seq_len = self.cur_seq_len + seq_len;
        tracing::debug!(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            previous_seq_len = old_seq_len,
            max_seq_len = self.max_seq_len,
            "updating autoregressive cache"
        );

        // Lazy initialization to match input tensor DType
        if self.cache.is_none() {
            // We use zeros to set the type, then cast if necessary.
            // On Flex, we must be careful with types.
            self.cache = Some(
                tensor
                    .clone()
                    .slice([0..batch_size, 0..num_heads, 0..1, 0..head_dim])
                    .repeat_dim(2, self.max_seq_len)
                    .mul_scalar(0.0), // Zero out but keep type/device
            );
        }
        let cache_tensor = self.cache.as_mut().unwrap();

        if new_seq_len > self.max_seq_len {
            // Sliding window: discard oldest tokens
            tracing::debug!(
                requested_seq_len = new_seq_len,
                max_seq_len = self.max_seq_len,
                "sliding autoregressive cache window"
            );
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = cache_tensor.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..head_dim,
            ]);
            *cache_tensor = cache_tensor.clone().slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    0..self.cur_seq_len,
                    0..head_dim,
                ],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        *cache_tensor = cache_tensor.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..head_dim,
            ],
            tensor,
        );

        self.cur_seq_len = new_seq_len;
        tracing::debug!(
            current_seq_len = self.cur_seq_len,
            "updated autoregressive cache"
        );

        if old_seq_len == 0 && seq_len == new_seq_len {
            return original_tensor;
        }

        cache_tensor.clone().slice([
            0..batch_size,
            0..num_heads,
            0..self.cur_seq_len,
            0..head_dim,
        ])
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_seq_len == 0
    }
}

/// Key-Value cache for a single transformer layer.
#[derive(Debug)]
pub struct KeyValueCache<B: Backend> {
    pub key: AutoregressiveCache<B>,
    pub value: AutoregressiveCache<B>,
}

impl<B: Backend> KeyValueCache<B> {
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, head_dim),
            value: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, head_dim),
        }
    }

    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }

    pub fn forward(
        &mut self,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (self.key.forward(key), self.value.forward(value))
    }

    pub fn len(&self) -> usize {
        self.key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.key.is_empty()
    }
}
