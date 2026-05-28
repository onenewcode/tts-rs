# Refactor Documentation Set

## Objective

Replace the legacy `tts_core` + `tts_qwen` architecture with a thinner,
model-first split that is easier to automate and safer to evolve:

- `tts_infer`: thin service-layer crate
- `tts_qwen3_tts`: concrete model crate for the current Qwen3-TTS model
- `tts_cli`: thin CLI over `tts_qwen3_tts`

This document set is intentionally decision-complete enough to drive later
Goal-mode implementation without further architectural invention.

## Design Principles

- Build one model correctly before extracting wider common abstractions.
- Delete false generic abstractions instead of renaming them.
- Split service lifecycle from model semantics.
- Keep package facts separate from request semantics and run-time options.
- Prefer resident-model design over request-scoped model loading.
- Keep public API smaller than internal lifecycle seams.

## Locked Decisions

- Delete `tts_core`.
- Delete `tts_qwen/src/arch`.
- Keep only three workspace crates: `tts_infer`, `tts_qwen3_tts`, `tts_cli`.
- `tts_infer` is a thin service layer, not a capability platform.
- `tts_qwen3_tts` owns package parsing, request semantics, backend selection,
  compiler logic, loaded-model state, and session execution.
- `tts_cli` is package-first and profile-subcommand driven.
- The public library API exposes only one-shot synthesis in v1.
- The internal service/model seam is already session-based in v1.

## Document Index

- `docs/refactor/01-crate-boundaries.md`
- `docs/refactor/02-tts-infer-contract.md`
- `docs/refactor/03-tts-qwen3-tts-design.md`
- `docs/refactor/04-cli-and-package-shape.md`
- `docs/refactor/05-migration-and-acceptance.md`
- `docs/refactor/06-package-manifest-example.md`
- `docs/refactor/07-source-tree-migration-map.md`

## Supporting Examples

- `docs/qwen3_tts_package.example.yaml`: target per-package manifest example
- `docs/models.example.yaml`: deprecated legacy registry marker; not a target
  design artifact
