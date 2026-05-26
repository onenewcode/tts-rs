# Qwen3-TTS Sampling Controls & Stopping Rules V5

## Summary

This document defines the fifth Rust inference milestone: replacing hard-coded greedy argmax token selection with configurable sampling and adding stopping rules.

Status: implemented (41 fast tests pass, V1-V4 backward-compatible greedy path unchanged).

**Implementation note**: `repetition_penalty` was folded into V5 rather than deferred.
The `SamplingConfig` includes `repetition_penalty: Option<f32>` and the penalty is applied
via gather/scatter-add in `generate_talker_tokens` and `generate_code_predictor_groups`.

V5 goals:

- add `SamplingConfig` with `temperature`, `top_k`, `top_p`, `do_sample`, `seed`, `repetition_penalty`
- add `StoppingRules` with `max_new_tokens`, `eos_token_id`
- add `suppress_tokens` filtering (matching Python's `suppress_tokens`)
- wire sampling into `generate_talker_tokens` and `generate_code_predictor_groups`
- keep greedy argmax as the default path (`do_sample = false`) for deterministic validation
- all sampling math stays inside Burn tensor/backend APIs

V5 non-goals:
- `min_new_tokens`
- beam search or contrastive search
- streaming token output
- `subtalker_*` parameter split (code predictor reuses the same `SamplingConfig`; independent configs deferred)

The expected output of this stage is a reusable `sample_token` function that subsumes the current `select_last_position_token` and supports both greedy and randomized modes, plus EOS-based early termination in the generation loops.

## Design

### New Types

```rust
/// Sampling mode: greedy argmax or randomized with temperature / top-k / top-p.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// false = greedy argmax (deterministic). true = apply temperature / top_k / top_p.
    pub do_sample: bool,
    /// Softmax temperature. Values near 0 approximate greedy. Default 1.0.
    pub temperature: f32,
    /// Keep only the top-k logits by magnitude before softmax. None = no truncation.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep the minimum set of tokens whose cumulative probability >= top_p.
    /// 1.0 = no truncation.
    pub top_p: f32,
    /// PRNG seed for reproducibility. None = non-deterministic (platform entropy).
    pub seed: Option<u64>,
}

/// Conditions that cause generation to stop before max_new_tokens.
#[derive(Debug, Clone)]
pub struct StoppingRules {
    /// Hard cap on generated tokens (prefill length not counted).
    pub max_new_tokens: usize,
    /// Stop early when this token is selected. None = no early termination.
    pub eos_token_id: Option<usize>,
}
```

### Core Function

```rust
/// Select one token per batch item from the last-position logits.
///
/// When `do_sample` is false, equivalent to greedy argmax.
/// When `do_sample` is true, applies temperature → top-k → top-p → softmax → multinomial.
///
/// Returns `(selected_token_ids, eos_mask)` where:
/// - `selected_token_ids`: shape `[batch, 1]`, the chosen token per batch item
/// - `eos_mask`: shape `[batch]`, true where EOS was selected (if eos_token_id is set)
pub fn sample_token<B: Backend>(
    logits: Tensor<B, 3>,
    sampling: &SamplingConfig,
    eos_token_id: Option<usize>,
    suppress_token_ids: &[usize],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Bool>)
```

### Sampling Algorithm (when `do_sample = true`)

Given logits `[batch, 1, vocab]` from the last position:

1. **Suppress tokens**: set logits of `suppress_token_ids` to `-inf`
2. **Temperature**: divide logits by `temperature` (clamped to `[1e-5, inf)`)
3. **Top-k**: find the k-th largest logit value per batch item; mask everything below it to `-inf`
4. **Top-p**: sort remaining logits descending; compute cumulative softmax probability; mask tokens beyond the first that exceeds `top_p` to `-inf`
5. **Softmax**: compute probabilities over the filtered logits
6. **Multinomial**: sample one token per batch item from the categorical distribution

Steps 3-4 match the PyTorch/HuggingFace `GenerationMixin` semantics: top-k is applied first, then top-p on the surviving tokens.

### Stopping Rules Integration

In `generate_talker_tokens`:

```
for step_idx in 0..max_new_tokens:
    if eos_token_id.is_some() && all_batch_items_have_eos(stopped_mask):
        break
    token = sample_token(...)
    mark batch items that selected EOS
    append token to output
```

The EOS token is checked **within the generation loop** (unlike Python's post-processing approach). This is cheaper because we save compute on already-stopped batch items.

### Suppress Tokens

Python uses `suppress_tokens = [vocab_size - 1024 .. vocab_size]` excluding `codec_eos_token_id`. This silences the trailing reserved token range. We add `suppress_token_ids: Vec<usize>` to `TalkerGenerateInput` with the same default derived from `vocab_size` and `codec_eos_token_id`.

### Config Changes

Add to `Qwen3TtsTalkerConfig` (from `config.json`):

```rust
pub codec_eos_token_id: usize,   // default 4198
pub codec_bos_token_id: usize,   // default 4197
pub codec_pad_token_id: usize,   // default 4196
```

These are read from `config.json`; the Python model stores them in `Qwen3TTSTalkerConfig`.

## Rust Interface Changes

### Modified: `TalkerGenerateInput`

Add fields:
- `sampling: SamplingConfig`
- `stopping: StoppingRules`
- `suppress_token_ids: Vec<usize>`

Remove field:
- `max_new_tokens` (moved into `StoppingRules`)

### Modified: `CodePredictorGenerateInput`

Add fields:
- `sampling: SamplingConfig`

No stopping rules needed — code predictor always generates exactly `num_code_groups - 1` tokens.

### Modified: `generate_talker_tokens`

Signature: add sampling/stopping/suppress parameters. Loop now checks EOS after each step.

### Modified: `generate_code_predictor_groups`

Signature: add sampling parameter. Replace greedy argmax with `sample_token`.

### New: `sample_token`

Public function in `inference.rs`. Works for both talker and code predictor.

## Backward Compatibility

Greedy mode (`do_sample = false`) must produce bit-identical results to the current `select_last_position_token`. The reference alignment test will use `do_sample = false` so V1-V4 validation is unaffected.

## Test Plan

Required fast unit tests:

- `sample_token` with `do_sample = false` produces same token as `select_last_position_token`
- `sample_token` with `temperature = 1e-5` (near-zero) approximates greedy
- `sample_token` with `top_k = 1` is equivalent to greedy (selects argmax)
- `sample_token` with `seed` is reproducible across calls
- `generate_talker_tokens` stops early when EOS is selected
- `generate_talker_tokens` does not exceed `max_new_tokens` even without EOS
- `generate_talker_tokens` with `do_sample = false` matches current greedy behavior
- `generate_code_predictor_groups` with `do_sample = false` matches current greedy behavior
- suppress tokens are never selected (probabilities are zero)

## Acceptance Criteria

- `cargo test -p tts_rs_qwen_burn` passes (all fast unit tests)
- `cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored` produces identical greedy tokens as before (V1-V4 unchanged)
- model code remains backend-portable Burn tensor/module code
- all public functions remain generic over `B: Backend`
- no `unsafe` code

## Next Stage

After V5 is stable:

1. add `repetition_penalty` (V6)
2. implement `audio_codec` decoder inference (V7)
3. connect generated talker codes to waveform decoding (V8)
4. decide whether to pull text-side preprocessing into Rust
