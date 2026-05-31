---
name: using-burn-tensor
description: Use when writing or reviewing Rust code that creates, reshapes, slices, moves, tests, or optimizes Burn tensors, especially when tensor rank, device placement, dtype behavior, autodiff boundaries, or synchronization cost may affect correctness or performance.
---

# Using Burn Tensor

## Overview

Burn tensor code stays correct when backend, rank, and tensor kind remain
explicit in the design. It stays fast when data remains on the target device,
work is expressed as tensor ops instead of host-side loops, and synchronization
points are treated as expensive boundaries.

Assume this skill applies any time a Rust change touches `Tensor<B, D, K>`,
`TensorData`, device moves, or Burn indexing and shape logic.

## When to Use

Use this skill when:

- implementing model math with `Tensor<B, D, K>`
- porting PyTorch, NumPy, or pseudocode into Burn
- reviewing Burn code for shape, device, or autodiff bugs
- debugging slow code that frequently reads tensors back to the CPU
- writing Burn tensor tests that need stable equality or approximate checks

Do not use this skill for:

- generic Rust collection code that does not involve Burn tensors
- backend authoring internals outside normal `Tensor` usage

## Core Mental Model

`Tensor<B, D, K>` has three independent axes of meaning:

| Part | Meaning | What to decide |
| --- | --- | --- |
| `B` | backend and device family | which backend owns execution |
| `D` | rank, not shape size | how many dimensions the tensor has |
| `K` | tensor kind | whether values are `Float`, `Int`, or `Bool` |

Rules:

- `D` is the number of dimensions, not the length of each dimension.
- The runtime shape is inferred from data creation and later transforms.
- `Float`, `Int`, and `Bool` are logical tensor kinds; the concrete element
  width is backend-defined, so do not hardcode assumptions like "float means
  f32 everywhere" unless the local backend contract says so.

## Non-Negotiable Rules

### 1. Create tensors on the device where they will be used

Prefer creating tensors directly on the target device with `from_data`,
`from_floats`, `zeros`, `ones`, `full`, `random`, or `arange`.

Good:

```rust
let device = B::Device::default();
let mask = Tensor::<B, 2, Bool>::from_data([[true, false]], &device);
```

Avoid patterns that create CPU-owned data repeatedly and move it inside hot
paths unless the move is truly part of the algorithm.

### 2. Keep computation on tensors, not in Rust loops

If the algorithm can be expressed with tensor ops, broadcasting, slicing,
selection, concatenation, reduction, or masking, prefer that over:

- `into_data()` inside the forward path
- per-element Rust loops
- repeated scalar extraction with `into_scalar()`

Host-side loops are for orchestration, not the math fast path.

### 3. Treat host reads as synchronization boundaries

Burn documents asynchronous execution for many backends. Reads such as
`into_data`, `into_scalar`, and explicit `sync` force synchronization. Some
other operations may also synchronize internally depending on backend behavior,
including certain device moves.

Therefore:

- do not call `into_data()` only to inspect intermediate values in hot code
- do not convert tensors to scalars in inner loops
- do not bounce between devices unless the boundary is intentional

If you need multiple reads, batch them with a `Transaction`.

```rust
let [logits, loss] = burn::tensor::Transaction::default()
    .register(logits)
    .register(loss)
    .execute()
    .try_into()
    .expect("two tensor payloads");
```

### 4. Rank errors are design errors

When writing Burn code, decide the rank first, then encode transforms that make
that rank obvious. Do not "poke at shapes until it compiles".

Good habits:

- write expected dimensions in comments near complex transforms
- use `dims()` when debugging shape flow
- keep batch, time, channel, and feature axes consistent across a module

### 5. Prefer Burn indexing primitives over manual extraction

For subranges and structured slicing:

- use `slice(...)`
- use the `s![]` macro for stepped or reversed slices
- use `select(dim, indices)` for gather-style index selection

This is usually clearer and keeps work on the tensor backend.

```rust
let recent = tokens.clone().slice(s![-256..]);
let picked = states.select(1, token_indices);
```

### 6. Make autodiff boundaries explicit

Use `require_grad()` only for tensors that must participate in gradient flow.
Use `detach()` when data must leave the current gradient graph, such as:

- cached activations reused as plain values
- batcher outputs that should not retain history
- metrics and logging paths

Do not rely on "autodiff probably does the right thing" when crossing module or
service boundaries.

## Preferred Workflow

When implementing tensor logic, follow this order:

1. Pick the rank and kind for each value.
2. Pick the target device and keep new tensors there.
3. Express shape changes with tensor transforms and indexing primitives.
4. Keep the forward path tensor-native.
5. Pull data back to the host only at explicit boundaries like tests, logging,
   serialization, or final output conversion.

## Pattern Catalog

### Creating tensors

Prefer:

- `Tensor::<B, D>::from_data(...)`
- `Tensor::<B, D>::from_floats(...)`
- `Tensor::<B, D>::zeros(...)`
- `Tensor::<B, D>::ones(...)`
- `Tensor::<B, D>::full(...)`
- `Tensor::<B, D>::random(...)`
- `Tensor::<B, 1, Int>::arange(...)`

Use `TensorData` when the test or boundary is about exact values and layout.

### Shape and indexing

Prefer:

- reshape and broadcast-oriented formulations over manual expansion loops
- `slice` or `s![]` for contiguous or stepped windows
- `select` for index-based selection
- `cat` when combining already prepared tensors in one intentional step

Avoid repeated tiny concatenations in a loop on the hot device. If many small
items are built elsewhere, assemble them in larger chunks before moving them to
the compute-critical device.

### Device movement

Prefer:

- a single clear ownership device per stage
- explicit `to_device(...)` boundaries
- one-way movement in a pipeline stage where possible

Avoid:

- alternating device moves in the same logical operation
- hidden host round-trips used only for inspection

### Testing and debugging

Prefer converting to `TensorData` in tests, then checking with:

- `assert_eq(..., strict)`
- `assert_approx_eq(...)`
- `assert_within_range(...)`

Good:

```rust
let actual = output.into_data();
let expected = burn::tensor::TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
actual.assert_approx_eq(&expected, burn::tensor::Tolerance::default());
```

Do not use exact float equality unless the test really depends on exact
bit-for-bit results.

## Before and After

### Bad: host-side inspection in the middle of tensor code

```rust
let hidden = model_forward(input);
let values = hidden.clone().into_data().to_vec::<f32>().unwrap();
let mean = values.iter().sum::<f32>() / values.len() as f32;
let mean = Tensor::<B, 1>::from_floats([mean], &device);
```

Problems:

- forces synchronization
- moves work to the CPU
- discards tensor backend optimizations

### Better: stay on the tensor backend

```rust
let hidden = model_forward(input);
let mean = hidden.mean();
```

### Bad: manual indexing logic in Rust

```rust
let ids = ids.into_data().to_vec::<i64>().unwrap();
let mut tail = Vec::new();
for id in ids.iter().skip(ids.len().saturating_sub(128)) {
    tail.push(*id);
}
let tail = Tensor::<B, 1, Int>::from_data(tail, &device);
```

### Better: slice on the tensor

```rust
let tail = ids.slice(s![-128..]);
```

## Common Mistakes

### "I only need `into_data()` for a quick check"

If this is inside a hot path, it is not quick. Use `dims()`, logging around the
boundary, or batch the read with a `Transaction`.

### "I will just pull one scalar at a time"

Repeated `into_scalar()` calls are a synchronization trap. Keep reductions on
the tensor side and materialize once at the boundary.

### "The compile-time dimension is the shape"

No. `D` is rank only. A `Tensor<B, 2>` can be `[2, 4]`, `[8, 16]`, or any
other two-dimensional shape.

### "Float means one fixed Rust primitive type"

Not necessarily. Burn abstracts logical tensor kind from backend element width.
Be careful when the surrounding code assumes a specific precision.

### "I can debug shape bugs by repeatedly moving tensors around"

That tends to hide the real issue and makes performance worse. Write the
intended dimension flow down and check tensor transforms against it.

## Review Checklist

When reviewing Burn tensor code, check these first:

- is the target `Tensor<B, D, K>` rank correct for the algorithm
- are new tensors created on the correct device
- does the forward path stay on tensor ops instead of host reads
- are `slice`, `s![]`, and `select` used instead of manual extraction
- are autodiff boundaries explicit with `require_grad` or `detach` where needed
- do tests compare through `TensorData` with appropriate tolerance

## References

Primary references used for this skill:

- Burn Book, Tensor:
  `https://burn.dev/books/burn/building-blocks/tensor.html`
- Burn Book, Asynchronous Execution:
  `https://burn.dev/books/burn/performance/good-practices/asynchronous-execution.html`
- Burn API docs, `Tensor`:
  `https://burn.dev/docs/burn/tensor/struct.Tensor.html`
- Burn API docs, `TensorData`:
  `https://burn.dev/docs/burn/tensor/struct.TensorData.html`
