# Asynchronous Execution

## What to assume

Many Burn backends execute asynchronously. Treat host reads and some device transitions as expensive boundaries, not as harmless inspections.

## Synchronization rules

Reads such as these force synchronization:

- `into_data()`
- `into_scalar()`
- explicit `sync`

Some `to_device(...)` transitions or backend-dependent operations may also synchronize internally.

Therefore:

- do not call `into_data()` only to inspect intermediate values in hot code
- do not convert tensors to scalars in inner loops
- do not bounce between devices unless the boundary is intentional
- do not assume a synchronous-looking API means the operation is cheap

## Batch host reads with `Transaction`

If a path must read multiple tensors for metrics, logging, or tests, batch the reads:

```rust
let [logits, loss] = burn::tensor::Transaction::default()
    .register(logits)
    .register(loss)
    .execute()
    .try_into()
    .expect("two tensor payloads");
```

Prefer one transaction over many separate host reads.

## Data loading and batching

Async execution also matters outside the model forward pass. If data augmentation, preprocessing, or batching work runs on the same device as the main computation, it can reduce throughput even when the math stays device-side.

Prefer:

- doing preprocessing on a separate device or backend when possible
- concatenating or batching items before moving them onto the main compute device
- doing one larger transfer instead of many tiny allocations

Less desirable for a hot path:

```rust
let items = load_many_small_tensors();
let batch = Tensor::cat(items, 0);
```

Better when preprocessing and training/inference devices differ:

```rust
let items = load_many_small_tensors();
let batch_cpu = Tensor::cat(items, 0);
let batch = Tensor::from_data(batch_cpu.into_data(), &device_training);
```

## Practical review questions

- Is a host read happening only for inspection?
- Is a scalar being pulled out inside a loop?
- Are many small device transfers replacing one clear batch transfer?
- Is preprocessing competing with the main compute device?

## Common mistake

### "I only need `into_data()` for a quick check"

Inside a hot path, it is not quick. Use `dims()`, log around the boundary, or batch reads with `Transaction`.

### "I will just pull one scalar at a time"

Repeated `into_scalar()` calls are a synchronization trap. Keep reductions on the tensor side and materialize once at the boundary.
