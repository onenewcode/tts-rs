# Kernel Fusion

## Goal

Write tensor code so Burn can keep operations fused where possible instead of forcing extra materialization, writes, or awkward execution boundaries.

## Fusion-friendly habits

Prefer:

- short-lived intermediates
- passing the latest value forward instead of cloning old values "just in case"
- grouping view operations before compute-heavy blocks
- keeping long element-wise chains tensor-native

Avoid:

- unnecessary `clone()` before reductions, matmuls, or other compute-bound ops
- scattering view operations through a compute block when they can be grouped
- extending the lifetime of intermediates that do not need to survive

## Be careful with view-style operations

These are often cheap individually, but they can reduce optimizer freedom when mixed awkwardly with compute:

- `slice`
- `slice_assign`
- `select`
- `gather`
- `scatter`
- `reshape`
- `swap_dims`
- `transpose`
- `unsqueeze`

Less fusion-friendly:

```rust
let out = tensor1.unsqueeze().matmul(tensor2) + tensor3.unsqueeze();
```

More fusion-friendly:

```rust
let tensor1 = tensor1.unsqueeze();
let tensor3 = tensor3.unsqueeze();
let out = tensor1.matmul(tensor2) + tensor3;
```

## Virtual vs materialized tensors

Fusion tradeoffs depend on whether an input is already concrete in memory or still behaves like a recent intermediate.

Heuristic:

- model parameters and long-lived stored tensors are often already materialized
- recent intermediate activations are often better treated as virtual

Implications:

- cloning a virtual tensor before a compute-bound op can force an extra write
- reordering around compute-bound ops may help more when inputs are recent intermediates
- when inputs are already concrete parameters, the obvious formulation may already be fine

If the tradeoff is ambiguous, profile the exact block.

## Practical review questions

- Are extra clones extending tensor lifetimes?
- Are view ops grouped coherently before the heavy compute?
- Is the code preserving a long tensor-native chain, or forcing early materialization?

## Common mistake

### "More clones make the code safer for performance"

Not in Burn. Extra clones can reduce fusion and force additional writes.

### "View operations are always free"

They are often cheap in isolation, but they can still weaken fusion opportunities when scattered through compute-heavy code.
