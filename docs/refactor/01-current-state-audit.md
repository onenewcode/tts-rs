# Current State Audit

## Summary

The current repository is functional for local Qwen3-TTS usage, but its
architecture is still shaped like a single-model application rather than a
local multi-driver framework.

The main refactor driver is not current correctness failure. The main driver is
boundary quality and future change cost.

## Observed Facts

### Workspace membership

Current workspace members:

- `tts_infer`
- `tts_qwen3_tts`
- `tts_cli`

Observed extra directory:

- `tts_core/` exists but is not part of the workspace

Implication:

- the repository already contains migration residue or an abandoned rename path
- architecture documents must explicitly identify the real source of truth

### Missing target-state docs

`AGENTS.md` references:

- `docs/architecture.md`
- `docs/testing_tts_qwen.md`
- `docs/refactor/`

These were missing from the observed repository state before this doc set was
added.

Implication:

- the repo had architectural intent but no checked-in target-state source of
  truth

### Thin core, thick driver

`tts_infer` is small and coherent:

- session engine abstraction
- model session contract
- audio output primitive
- lifecycle guardrails

`tts_qwen3_tts` is overloaded:

- public request types
- public façade
- package normalization
- config loading
- request compilation
- runtime sampling helpers
- model graph code
- weight loading
- speaker encoder logic
- codec decode
- profiling support

Implication:

- the current crate is acting as both product surface and driver internals

### CLI still owns semantic assembly

`tts_cli/src/cli.rs` currently performs:

- request construction
- voice-clone input validation
- package source selection
- backend defaulting

Implication:

- the shell still owns business semantics that should move down into framework
  or driver-facing service layers

## Current Architectural Risks

### 1. Qwen3 crate boundary is too broad

Risk:

- every change in public request semantics, load configuration, runtime
  execution, or graph implementation hits the same crate surface

Impact:

- hard to reason about stable versus volatile boundaries
- future additional model drivers have no clean architectural reference

### 2. Framework layer is missing

Risk:

- there is no explicit local model-service core

Impact:

- no canonical place for driver registration
- no canonical place for loaded instance lifecycle
- no canonical place for common diagnostics

### 3. CLI is still doing too much

Risk:

- frontend shell behavior and synthesis semantics are coupled

Impact:

- future non-CLI frontends will have to re-implement request assembly rules

### 4. Documentation and code drift

Risk:

- repo guidance references docs that do not exist

Impact:

- engineers cannot reliably distinguish current state from intended state

## Refactor Implications

The audit supports these conclusions:

- a framework core should be introduced before attempting aggressive model-
  internal crate splits
- `tts_qwen3_tts` should stay single-crate initially, but be strongly layered
  internally
- lifecycle, capability inspection, and diagnostics need first-class homes
- documentation must be treated as a concrete deliverable, not as a follow-up

## Acceptance Criteria

This audit is acceptable only if it clearly identifies:

- the current workspace truth
- the stale `tts_core/` directory situation
- the missing-doc problem
- the Qwen3 crate overload problem
- the CLI semantic ownership problem

