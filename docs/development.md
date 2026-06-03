# Development Guide

## Summary

Use this guide when changing code in `tts-rs`. It defines the default
development flow, the current workspace boundaries, and the Burn tensor rules
that matter most in `tts_qwen3_tts`.

This document complements:

- `docs/architecture.md` for crate and runtime boundaries
- `docs/TEST.md` for tests, fast checks, and smoke procedures
- `docs/using-burn-tensor/SKILL.md` for Burn tensor best practices

## Required Checks

Every code change must pass:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

Use `docs/TEST.md` for the smallest relevant test or smoke path.

## Default Workflow

1. Check `docs/architecture.md` before moving logic across crates or layers.
2. Keep changes local first; extract helpers only when reuse is real.
3. Run `cargo fmt --all --check`.
4. Run `cargo clippy --workspace --all-targets`.
5. Run the relevant verification from `docs/TEST.md`.

## Workspace Boundaries

Keep responsibilities split as follows:

- `tts_infer` (`tts_core`) owns model lifecycle, handles, capabilities, and
  shared audio/result primitives
- `tts_error` owns shared diagnostics
- `tts_qwen3_tts` owns Qwen3-specific loading, execution, and model internals
- `tts_app` owns shell-to-driver orchestration
- `tts_cli` stays a thin command-line adapter over `tts_app`

Do not move model-private logic into `tts_cli`, and do not put CLI semantics
into `tts_qwen3_tts`.

## Verification Expectations By Change Type

### API or request-shape changes

- Re-check the public request surface in `tts_qwen3_tts`
- Re-run the relevant checks from `docs/TEST.md`
- Keep `tts_cli` as an adapter, not a business-logic layer

### Runtime or model execution changes

- Re-check tensor device and dtype placement
- Re-run `cargo clippy --workspace --all-targets`
- Run at least one relevant model-backed smoke path from `docs/TEST.md`

### CLI-only changes

- Keep logic in `tts_app` when it affects request preparation
- Keep `tts_cli` focused on parsing, invoking services, and reporting results

## Burn Tensor Rules

When changing `tts_qwen3_tts`, follow `docs/using-burn-tensor/SKILL.md`. In
this repository:

- create tensors on the device and in the dtype where they are consumed
- keep the hot path tensor-native; avoid host reads in generation and model
  compute paths
- use host reads only at explicit boundaries such as final audio extraction,
  scalar decisions, logging, serialization, or tests
- keep reshape, slice, transpose, and broadcast steps close to the compute they
  support
- keep dtype conversions only where they are required by numeric stability,
  quantization boundaries, rotary/position math, or explicit output conversion

- prefer building tensors directly in the target dtype instead of creating
  temporary `f32` tensors and casting later
- prefer direct `cast(...)` when conversion is needed; do not add manual
  same-dtype guards just to avoid a no-op cast
- keep `dequantize()` only where the following operation truly requires float
  math or host extraction
- keep `try_into_data()` and `try_into_scalar()` out of hot paths

If a performance conclusion is uncertain, profile the exact path instead of
guessing from style alone.

## Code Review Checklist

Before finishing a change, check:

- does the crate boundary still make sense?
- is any new helper actually shared enough to justify extraction?
- are tensor creation, dtype conversion, and host synchronization minimal?
- did the change keep `tts_cli` thin?
- did `cargo fmt --all --check` pass?
- did `cargo clippy --workspace --all-targets` pass?

## Verification Entry Points

Default verification entry points:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

For test targets and smoke commands, use `docs/TEST.md` as the authoritative
source instead of duplicating command lists here.
