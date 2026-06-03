# Testing And Review

## Testing

Prefer converting to `TensorData` at the test boundary, then assert through the comparison helpers.

Good:

```rust
let actual = output.into_data();
let expected = burn::tensor::TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
actual.assert_approx_eq(&expected, burn::tensor::Tolerance::default());
```

Prefer:

- `assert_eq(..., strict)` when exact layout is the point
- `assert_approx_eq(...)` for floating-point outputs
- `assert_within_range(...)` when a range is the actual contract

Do not use exact float equality unless the test truly depends on bit-for-bit results.

## Debugging

Prefer:

- writing down the intended dimension flow
- checking `dims()` before pulling tensors back to host
- confirming device ownership at module boundaries

Avoid debugging shape problems by repeatedly moving tensors across devices or converting them to host data in the hot path.

## Review checklist

Check these first:

- Is the intended `Tensor<B, D, K>` rank correct for the algorithm?
- Are new tensors created on the correct device?
- Does the hot path stay on tensor ops instead of host reads?
- Are `into_data`, `into_scalar`, `sync`, or `to_device` used in places that likely force avoidable synchronization?
- Are indexing and reshaping expressed with Burn primitives instead of manual extraction?
- Are batching and concatenation happening on the right side of the device boundary?
- Are intermediate clones extending tensor lifetimes and hurting fusion?
- Are hot shapes obviously awkward when they could be regularized?
- Are autodiff boundaries explicit with `require_grad()` or `detach()` where needed?

## Common mistakes

### "The compile-time dimension is the shape"

No. `D` is rank only. A `Tensor<B, 2>` can be `[2, 4]`, `[8, 16]`, or any other two-dimensional shape.

### "Float means one fixed Rust primitive type"

Not necessarily. Burn abstracts logical kind from concrete backend precision.

### "I can debug shape bugs by repeatedly moving tensors around"

That tends to hide the real issue and make performance worse. Write the intended dimension flow down and check the transforms directly.
