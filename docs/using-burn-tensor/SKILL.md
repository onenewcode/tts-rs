---
name: using-burn-tensor
description: Write, review, debug, or optimize Rust code that uses Burn tensors (`Tensor<B, D, K>`), especially when shape/rank, device placement, dtype behavior, host synchronization, kernel fusion, kernel selection, batching, or autodiff boundaries may affect correctness or performance.
---

# Using Burn Tensor

Use this skill to keep Burn tensor code correct first, then fast.

## Core workflow

1. Decide rank, tensor kind, and owning device before writing math.
2. Keep the forward path tensor-native; avoid host reads in hot code.
3. Group shape/view operations coherently so fusion remains possible.
4. Prefer backend-friendly shapes when performance matters.
5. Read back to host only at explicit boundaries such as tests, logging, serialization, or final output conversion.

## Read these references only when needed

- Read `references/core-rules.md` for the baseline rules on rank, device placement, indexing, and autodiff boundaries.
- Read `references/asynchronous-execution.md` when reviewing `into_data`, `into_scalar`, `sync`, `to_device`, batching, data loading, or CPU/GPU handoff behavior.
- Read `references/kernel-fusion.md` when optimizing chains of tensor ops, investigating extra clones, or deciding how to order reshape/slice/transpose work around compute-heavy ops.
- Read `references/kernel-selection.md` when performance depends on tensor shapes, cold-start autotune behavior, or backend-specific kernel choice.
- Read `references/testing-and-review.md` when writing tests, debugging shape problems, or doing a code review pass.

## Default review order

1. Confirm the intended `Tensor<B, D, K>` rank and semantic axes.
2. Confirm tensors are created on the device where they are consumed.
3. Confirm the hot path stays on tensor ops instead of repeated host reads.
4. Confirm indexing and reshaping use Burn primitives instead of manual extraction.
5. Confirm performance-sensitive code is not accidentally blocking async execution or weakening fusion.

## Escalation rule

If performance conclusions are ambiguous, profile the exact block instead of guessing from style alone.
