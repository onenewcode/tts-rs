# Migration Plan

## Summary

The refactor should proceed in bounded phases. The goal is not to rewrite the
repository in one jump. The goal is to establish stable boundaries in the right
order.

## Phase 1: Documentation Baseline

Deliverables:

- `docs/architecture.md`
- `docs/testing_tts_qwen.md`
- `docs/refactor/01-current-state-audit.md`
- `docs/refactor/02-target-architecture.md`
- `docs/refactor/03-migration-plan.md`
- `docs/refactor/04-api-spec.md`

Exit criteria:

- all referenced architecture docs exist
- docs name the target core objects and state model
- docs explain why Qwen3 remains single-crate initially

## Phase 2: Core Rename And Scope Freeze

Actions:

- promote the current `tts_infer` responsibility set into future `tts_core`
- decide rename strategy without broadening scope prematurely
- freeze what is and is not part of the core public contract

Exit criteria:

- lifecycle/session/audio primitives are still coherent
- no Qwen3 request semantics are moved into core
- stale root `tts_core/` directory situation is resolved or explicitly parked

## Phase 3: Shared Diagnostics Foundation

Actions:

- add `tts_error`
- move shared categories/codes/diagnostic containers there
- keep concrete model errors in model crates

Exit criteria:

- shared diagnostics no longer need to live inside model crates
- model-specific error enums still remain model-specific

## Phase 4: Qwen3 Internal Layering

Actions:

- reorganize `tts_qwen3_tts` into:
  - `surface`
  - `loading`
  - `capabilities`
  - `execution`
  - `backend`
- move current compiler responsibilities into `execution`
- move request materialization logic into `execution`

Exit criteria:

- `surface` contains only public driver-facing types and façade
- `loading` owns package normalization and instance construction
- `capabilities` is a first-class subsystem
- execution internals are no longer surfaced through crate root structure

## Phase 5: Framework Driver Registration

Actions:

- introduce `DriverDescriptor`
- introduce `DriverRegistry`
- register Qwen3 as the first explicit driver

Exit criteria:

- a driver is no longer just "the crate the CLI knows about"
- registered driver metadata is queryable without loading an instance

## Phase 6: Loaded Instance Management

Actions:

- introduce `LoadedModelHandle`
- introduce `ModelManager`
- implement `Ready / Busy / Closed`
- implement close/remove semantics

Exit criteria:

- loaded instances have framework-assigned IDs
- same-instance execution is serialized by the handle
- manager removes only closed instances

## Phase 7: Application Service Layer

Actions:

- add `tts_app`
- move orchestration and request assembly out of CLI

Exit criteria:

- CLI is not constructing rich driver requests directly
- application services can be reused by future local frontends

## Phase 8: CLI Thinning

Actions:

- keep only shell parsing and output/reporting in `tts_cli`

Exit criteria:

- CLI tests are shell-focused only
- synthesis semantics are no longer owned by the CLI crate

## Phase 9: Final Cleanup

Actions:

- remove dead migration artifacts
- align README and AGENTS with the implemented architecture

Exit criteria:

- docs and code no longer disagree on the crate map
- no stale architectural directory remains unexplained

## Verification Gates

Every phase must satisfy both of these gates before the next phase begins.

### Design gate

The changed boundary must be explainable in one paragraph without referring to
implementation trivia.

### Verification gate

At minimum:

```bash
cargo test -p tts_infer
cargo test -p tts_qwen3_tts --test public_surface
cargo test -p tts_qwen3_tts --test compiler_load
cargo test -p tts_cli --test cli_parse
```

## Acceptance Criteria

This migration plan is acceptable only if it defines:

- multiple phases
- concrete exit criteria for each phase
- verification gates
- the order in which architecture boundaries should be established

