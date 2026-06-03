# Core Rules

## Mental model

`Tensor<B, D, K>` has three independent axes of meaning:

| Part | Meaning | What to decide |
| --- | --- | --- |
| `B` | backend and device family | which backend owns execution |
| `D` | rank, not shape size | how many dimensions the tensor has |
| `K` | tensor kind | whether values are `Float`, `Int`, or `Bool` |

Rules:

- `D` is the number of dimensions, not the runtime extent of each dimension.
- Runtime shape comes from creation plus later transforms.
- `Float`, `Int`, and `Bool` are logical kinds; concrete precision is backend-defined.

## Non-negotiable rules

### Create tensors on the device where they will be used

Prefer `from_data`, `from_floats`, `zeros`, `ones`, `full`, `random`, and `arange` on the target device.

```rust
let device = B::Device::default();
let mask = Tensor::<B, 2, Bool>::from_data([[true, false]], &device);
```

Avoid repeatedly creating CPU-side data and moving it into a hot path unless that transfer is part of the algorithm.

### Keep computation on tensors, not in Rust loops

If the algorithm can be expressed with tensor ops, broadcasting, slicing, masking, selection, concatenation, or reduction, prefer that over:

- `into_data()` in the forward path
- per-element Rust loops
- repeated `into_scalar()` calls

Host loops are for orchestration, not the math fast path.

### Treat rank errors as design errors

Decide the rank first, then write transforms that make the rank obvious.

Good habits:

- write expected dimensions near complex transforms
- use `dims()` when debugging shape flow
- keep batch, time, channel, and feature axes consistent within a module

### Prefer Burn indexing primitives

For subranges and structured slicing, prefer:

- `slice(...)`
- `s![]` for stepped or reversed slices
- `select(dim, indices)` for gather-style selection

```rust
let recent = tokens.clone().slice(s![-256..]);
let picked = states.select(1, token_indices);
```

### Make autodiff boundaries explicit

Use `require_grad()` only where gradient flow is required. Use `detach()` when data must leave the current graph, such as:

- cached activations reused as plain values
- batcher outputs that should not retain history
- metrics and logging paths

## Pattern catalog

### Creating tensors

Prefer:

- `Tensor::<B, D>::from_data(...)`
- `Tensor::<B, D>::from_floats(...)`
- `Tensor::<B, D>::zeros(...)`
- `Tensor::<B, D>::ones(...)`
- `Tensor::<B, D>::full(...)`
- `Tensor::<B, D>::random(...)`
- `Tensor::<B, 1, Int>::arange(...)`

Use `TensorData` when the boundary is about exact values and layout.

### Shape and indexing

Prefer:

- reshape- and broadcast-oriented formulations over manual expansion loops
- `slice` or `s![]` for contiguous or stepped windows
- `select` for index-based selection
- `cat` when combining already prepared tensors in one intentional step

Avoid repeated tiny concatenations in a loop on the hot device. Assemble larger chunks before moving them to the compute-critical device.
