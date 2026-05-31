# Qwen3 TTS Model Burn Tensor Refactor Design

## Goal

Refactor `tts_qwen3_tts/src/model` so shared tensor/shape logic is centralized,
internal tensor flows follow the guidance in `docs/burn-tensor-skill.md`, and
host-read boundaries are explicit instead of being scattered across the model
implementation.

## Scope

In scope:

- `tts_qwen3_tts/src/model/nn`
- `tts_qwen3_tts/src/model/talker`
- `tts_qwen3_tts/src/model/codec`
- `tts_qwen3_tts/src/model/speaker`
- `tts_qwen3_tts/src/model/mod.rs`

Out of scope:

- crates outside `tts_qwen3_tts/src/model`
- rewriting speaker DSP/FFT work into Burn tensors
- changing public crate APIs unless the refactor requires a small internal-only
  signature cleanup

## Current Problems

### Shared tensor logic is fragmented

`tts_qwen3_tts/src/model/nn` already contains small helpers, but common tensor
operations still live inside larger feature files:

- last-step selection is split between `nn/sequence.rs` and
  `talker/infer.rs`
- repeated flatten/unflatten logic is duplicated in `talker/network.rs` and
  `codec/model.rs`
- device-local index tensor construction appears as ad hoc `TensorData::new`
  blocks in multiple files

This makes shape conventions harder to audit and increases the risk of subtle
rank/layout drift.

### Some host reads are still inside model logic

Most `try_into_data` uses are valid output boundaries, but two classes still
need cleanup:

- generation-path scalar inspection in `talker/infer.rs`
- codec token materialization in `codec/model.rs`

These are the main places where `burn-tensor-skill.md` flags unnecessary
synchronization risk.

### Burn-native shape/indexing patterns are not consistently applied

The model already uses `reshape`, `select`, `repeat_dim`, `cat`, and `scatter`,
but the style is inconsistent. Several files manually reconstruct one-off index
or shape flows instead of relying on shared helpers that make rank expectations
explicit.

## Design Principles

1. Keep computation on Burn tensors until an explicit API boundary.
2. Treat `try_into_data` / `try_into_scalar` as named boundaries, not inline
   implementation details.
3. Centralize reusable shape/index helpers in `tts_qwen3_tts/src/model/nn`.
4. Preserve behavior and tensor layout unless a mismatch with Burn guidance is
   being corrected.
5. Do not force CPU-heavy DSP code into Burn if the tensor boundary is already
   well placed.

## Approaches Considered

### Approach A: Shared helper layer first, then migrate subsystems

Add a new shared tensor helper module in `tts_qwen3_tts/src/model/nn`, migrate
`talker` to it first, then fold `codec` and `speaker` into the same patterns.

Pros:

- creates a stable vocabulary for rank/shape operations
- reduces duplicated fixes across `talker` and `codec`
- keeps risk localized and testable by stage

Cons:

- requires touching helper code before feature code
- adds one more internal module to maintain

### Approach B: Refactor each subsystem independently

Leave `nn` mostly unchanged and rewrite `talker`, `codec`, and `speaker` in
place.

Pros:

- direct path to each hotspot
- no new common module to design up front

Cons:

- duplicates fixes
- makes it harder to prove the repo converges on one Burn style
- increases the chance that `talker` and `codec` drift again later

### Approach C: Massive one-pass rewrite of all tensor flows

Apply all helper extraction and tensor cleanup in one large change.

Pros:

- fewer intermediate states

Cons:

- much harder to review and debug
- weak failure isolation
- high regression risk in model-heavy code

### Recommendation

Use Approach A.

It best matches the user goal of both extracting common pieces and correcting
Tensor usage under the Burn guidance. It also creates a clear sequence:
shared layer -> talker -> codec -> speaker.

## Target Architecture

### 1. Shared tensor helper module

Add a new module under `tts_qwen3_tts/src/model/nn` dedicated to small,
backend-safe tensor helpers.

Responsibilities:

- select the last sequence step while preserving rank expectations
- flatten `[batch, seq, hidden] -> [batch * seq, hidden]`
- restore `[batch * seq, hidden] -> [batch, seq, hidden]`
- build small device-local integer tensors for `select` / index operations
- wrap explicit readback boundaries used by model internals

Non-responsibilities:

- no feature-specific sampling, attention, or codec logic
- no business-level error messages outside boundary helpers

### 2. Talker cleanup

Primary files:

- `tts_qwen3_tts/src/model/talker/infer.rs`
- `tts_qwen3_tts/src/model/talker/network.rs`
- `tts_qwen3_tts/src/model/talker/sampling.rs`

Changes:

- route last-step selection through shared helpers
- replace inline scalar inspection with a named boundary helper
- normalize repeated flatten/unflatten logic for LM head application
- keep generation-path tensor work on device except for explicit EOS decisions

### 3. Codec cleanup

Primary files:

- `tts_qwen3_tts/src/model/codec/model.rs`
- `tts_qwen3_tts/src/model/codec/runtime/decode.rs`

Changes:

- extract repeated flatten/unflatten and head-shape operations into shared
  helpers where that improves clarity
- make the codec token readback path explicit and isolated
- keep valid waveform export readback in `runtime/decode.rs` as an output
  boundary helper, not a hidden hot-path read

### 4. Speaker cleanup

Primary files:

- `tts_qwen3_tts/src/model/speaker/feature.rs`
- `tts_qwen3_tts/src/model/speaker/infer.rs`

Changes:

- keep STFT / mel logic CPU-side
- make the final tensor creation and embedding export follow the same explicit
  boundary pattern as other model outputs

## Data Flow Changes

### Last-step extraction

Current state:

- some callers use `select_last_sequence_step`
- some callers immediately reshape again or wrap another local helper

Target state:

- one shared helper for rank-3 last-step extraction
- one shared helper for rank-2 "last hidden state" extraction used by talker

### Flatten / unflatten conventions

Current state:

- multiple files locally compute `batch * seq`
- reshape conventions are repeated with slightly different local naming

Target state:

- shared helpers encode the intended rank changes
- callers still supply dimensions, but the operation names explain intent

### Host read boundaries

Current state:

- valid export boundaries and questionable inline reads use the same low-level
  APIs directly

Target state:

- output/export boundaries continue to read back data
- internal decision points use small named helpers so synchronization cost is
  easy to audit and move later if Burn gains a better alternative

## Error Handling

- Keep existing `QwenTtsInferenceError` / `Qwen3TtsInferenceError` flows.
- New boundary helpers should map readback failures into the existing tensor
  read variants with precise path-specific messages.
- Helper modules should stay small and avoid inventing a parallel error type.

## Testing Strategy

### Shared helper tests

Add unit tests next to the new `nn` helper module for:

- last-step selection
- flatten/unflatten shape round trips
- device-local index tensor creation where deterministic

### Talker regression tests

Preserve or add tests for:

- EOS detection behavior
- sampling suppression behavior
- last hidden state extraction

### Codec regression tests

Preserve or add tests for:

- codec token layout stability after helper extraction
- waveform export boundary behavior

### Speaker regression tests

Preserve or add tests for:

- embedding export shape/data path
- mel tensor output shape from the CPU feature extractor

## Rollout Order

1. Add shared tensor helper module and tests.
2. Migrate `talker` to the shared helper layer.
3. Migrate `codec` to the shared helper layer and isolate codec readback.
4. Normalize `speaker` boundary helpers.
5. Run focused tests for `nn`, `talker`, `codec`, and `speaker`.
6. Do a final search for direct internal `try_into_data` / `try_into_scalar`
   uses and confirm the remaining ones are legitimate boundaries.

## Acceptance Criteria

The refactor is complete when all of the following are true:

- `tts_qwen3_tts/src/model` has a single obvious home for common tensor/shape
  helpers
- `talker` and `codec` no longer duplicate the main last-step and
  flatten/unflatten patterns unnecessarily
- internal tensor readbacks are reduced and named; remaining readbacks are
  explicit output or boundary operations
- tests cover the new helpers and the most important behavior-preserving paths
- remaining tensor code is consistent with the guidance in
  `docs/burn-tensor-skill.md`
