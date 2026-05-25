# Qwen3-TTS Repetition Penalty V6

## Summary

This document defines the repetition penalty feature (folded into V5 `SamplingConfig`).
Repetition penalty discourages or encourages token reuse during autoregressive generation
by scaling logits of previously-selected tokens.

Status: implemented.

## Design

### Algorithm

For each past token `t` in generation history, modify the corresponding logit:

    logits[:, t] /= penalty

When `penalty < 1.0`, repeated tokens are discouraged. When `penalty > 1.0`, they are
encouraged. `penalty == 1.0` is a no-op. Python defaults to `1.05`.

### Burn Implementation

Burn's `scatter` only supports `IndexingUpdateOp::Add`, so we use the delta trick:

    delta = logit * (1/penalty - 1)
    logits.scatter_add(past_ids, delta)

Steps:
1. **Gather** logit values at past token positions via `logits.gather(1, past_ids)`
2. **Scale** by `(1/penalty - 1)` to compute deltas
3. **Scatter-add** deltas back to logits: `logits.scatter(1, past_ids, deltas, Add)`

All operations stay on-device; no host-side copies.

### Integration Points

| Function | When applied |
|---|---|
| `generate_talker_tokens` | Before each `sample_token` call, using accumulated token history |
| `generate_code_predictor_groups` | Before each `sample_token` call in the autoregressive loop |
| `generate_talker_tokens` (prefill) | No history → passes empty `[batch, 0]` tensor |

### Rust Interface

```rust
pub struct SamplingConfig {
    // ... existing fields ...
    pub repetition_penalty: Option<f32>,  // None = off
}
```

Helper:
```rust
fn apply_repetition_penalty_3d<B: Backend>(
    logits: Tensor<B, 3>,
    past_token_ids: &Tensor<B, 2, Int>,
    penalty: Option<f32>,
) -> Tensor<B, 3>
```

## Python Reference Alignment

The Python model passes `repetition_penalty` to HuggingFace `GenerationMixin.generate()`.
To validate Rust behavior:

1. Generate Python tokens with `repetition_penalty=1.0` (off) and `repetition_penalty=1.05` (default)
2. Run Rust generation with matching penalty values
3. Compare token sequences: with `repetition_penalty=1.0`, both should produce identical greedy tokens
4. With `repetition_penalty=1.05`, tokens should match within BF16 precision

### Reference Data

- `reference_v6_penalty_off.json`: penalty=1.0, should match V3 greedy tokens
- `reference_v6_penalty_on.json`: penalty=1.2 (amplified for visibility), per-step logits + tokens

## Test Plan

Required tests (all implemented in `talker/tests.rs`):

- `sample_token` with `repetition_penalty=None` = no-op ✓
- All existing generation tests pass with `repetition_penalty=None` ✓
- (future) `generate_talker_tokens` with `repetition_penalty=1.2` produces different tokens than `=1.0`

## Acceptance Criteria

- `cargo test -p tts_rs_qwen_burn` passes (41 tests) ✓
- Greedy mode with `repetition_penalty=None` is bit-identical to pre-V5 behavior ✓
- Model code remains backend-portable ✓
