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

    /// Reset the cache state while keeping any allocated backing tensor.
    pub fn reset_preserve_allocation(&mut self) {
        tracing::debug!(
            previous_seq_len = self.cur_seq_len,
            max_seq_len = self.max_seq_len,
            "reset autoregressive cache length"
        );
        self.cur_seq_len = 0;
    }

    /// Update the cache and return the current valid slice.
    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();
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

        if self.cache.is_none() {
            let device = tensor.device();
            let dtype = tensor.dtype();
            self.cache = Some(Tensor::<B, 4>::zeros(
                [self.max_batch_size, num_heads, self.max_seq_len, head_dim],
                (&device, dtype),
            ));
        }
        let mut cache_tensor = self.cache.take().expect("cache must be initialized");

        if seq_len >= self.max_seq_len {
            tracing::debug!(
                seq_len,
                max_seq_len = self.max_seq_len,
                "incoming cache step exceeds window; keeping latest slice only"
            );
            let keep_from = seq_len - self.max_seq_len;
            let latest =
                tensor.slice([0..batch_size, 0..num_heads, keep_from..seq_len, 0..head_dim]);
            cache_tensor = cache_tensor.slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    0..self.max_seq_len,
                    0..head_dim,
                ],
                latest.clone(),
            );
            self.cur_seq_len = self.max_seq_len;
            self.cache = Some(cache_tensor);
            return latest;
        }

        if new_seq_len > self.max_seq_len {
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
            cache_tensor = cache_tensor.slice_assign(
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

        let original_tensor = (old_seq_len == 0 && seq_len == new_seq_len).then(|| tensor.clone());
        cache_tensor = cache_tensor.slice_assign(
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

        if let Some(original_tensor) = original_tensor {
            self.cache = Some(cache_tensor);
            return original_tensor;
        }

        let current = cache_tensor.clone().slice([
            0..batch_size,
            0..num_heads,
            0..self.cur_seq_len,
            0..head_dim,
        ]);
        self.cache = Some(cache_tensor);
        current
    }

    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}

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

    pub fn reset_preserve_allocation(&mut self) {
        self.key.reset_preserve_allocation();
        self.value.reset_preserve_allocation();
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
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::KeyValueCache;
    use crate::loading::runtime::RuntimeBackend;

    #[test]
    fn reset_preserve_allocation_clears_len_without_dropping_cache_tensors() {
        let device = Default::default();
        let mut cache = KeyValueCache::<RuntimeBackend>::new(1, 1, 4, 2);
        let key = Tensor::<RuntimeBackend, 4>::ones([1, 1, 2, 2], &device);
        let value = Tensor::<RuntimeBackend, 4>::ones([1, 1, 2, 2], &device);

        let _ = cache.forward(key, value);
        assert_eq!(cache.len(), 2);
        assert!(cache.key.cache.is_some());
        assert!(cache.value.cache.is_some());

        cache.reset_preserve_allocation();

        assert_eq!(cache.len(), 0);
        assert!(cache.key.cache.is_some());
        assert!(cache.value.cache.is_some());
    }

    #[test]
    fn reset_preserve_allocation_exposes_only_new_forward_values() {
        let device = Default::default();
        let mut cache = KeyValueCache::<RuntimeBackend>::new(1, 1, 4, 1);
        let first_key = Tensor::<RuntimeBackend, 1>::from_floats([1.0, 2.0].as_slice(), &device)
            .reshape([1, 1, 2, 1]);
        let first_value = Tensor::<RuntimeBackend, 1>::from_floats([3.0, 4.0].as_slice(), &device)
            .reshape([1, 1, 2, 1]);

        let _ = cache.forward(first_key, first_value);
        cache.reset_preserve_allocation();

        let next_key = Tensor::<RuntimeBackend, 1>::from_floats([9.0].as_slice(), &device)
            .reshape([1, 1, 1, 1]);
        let next_value = Tensor::<RuntimeBackend, 1>::from_floats([8.0].as_slice(), &device)
            .reshape([1, 1, 1, 1]);
        let (key_out, value_out) = cache.forward(next_key, next_value);

        let key_values = key_out
            .try_into_data()
            .expect("key cache output should be readable")
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("key cache output should convert to vec");
        let value_values = value_out
            .try_into_data()
            .expect("value cache output should be readable")
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("value cache output should convert to vec");

        assert_eq!(cache.len(), 1);
        assert_eq!(key_values, vec![9.0]);
        assert_eq!(value_values, vec![8.0]);
    }
}
