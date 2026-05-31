# Qwen3 Model Structure Refactor Implementation Plan

## Goal

Make `tts_qwen3_tts/src/model` a pure model-implementation tree and move
loaded-model runtime assembly into `tts_qwen3_tts/src/execution`, while also
making `speaker/`, `codec/`, and `talker/` structurally aligned.

## Files

- Create: `docs/superpowers/specs/2026-05-31-qwen3-model-structure-refactor-design.md`
- Create: `docs/superpowers/plans/2026-05-31-qwen3-model-structure-refactor.md`
- Create: `tts_qwen3_tts/src/execution/audio_finalize.rs`
- Create: `tts_qwen3_tts/src/execution/backend_runtime.rs`
- Create: `tts_qwen3_tts/src/execution/loaded_model.rs`
- Modify: `tts_qwen3_tts/src/execution/mod.rs`
- Modify: `tts_qwen3_tts/src/loading/mod.rs`
- Modify: `tts_qwen3_tts/src/capabilities/mod.rs`
- Modify: `tts_qwen3_tts/src/model/mod.rs`
- Restructure: `tts_qwen3_tts/src/model/speaker/**/*`
- Restructure: `tts_qwen3_tts/src/model/codec/**/*`
- Restructure: `tts_qwen3_tts/src/model/talker/**/*`
- Delete: `tts_qwen3_tts/src/model/runtime.rs`

## Steps

1. Add the design document and this implementation plan before touching code.
2. Copy loaded-model/session/backend logic out of `model/mod.rs` into new
   `execution` modules.
3. Move reference codec prefix helpers into `execution/audio_finalize.rs` and
   keep their unit tests there.
4. Update `loading`, `capabilities`, and `execution/mod.rs` to use the new
   `execution::loaded_model::Qwen3TtsLoadedModel` path.
5. Restructure `speaker/`, `codec/`, and `talker/` so each subtree follows the
   same top-level split: `config.rs`, `infer/`, `network/`, `weights.rs`.
6. Split oversized `network/mod.rs` files by operator family instead of keeping
   monolithic network definitions.
7. Reduce `model/mod.rs` to model-private module declarations only.
8. Delete `model/runtime.rs`.
9. Run formatting, tests, and `clippy -D warnings`.

## Verification

Run:

```bash
cargo fmt --check
cargo test -p tts_qwen3_tts --all-targets --all-features
cargo clippy -p tts_qwen3_tts --all-targets --all-features -- -D warnings
```

## Risks To Check

- missing private-path updates after module migration
- partial subtree alignment where only one model family uses nested `infer/` or
  `network/`
- feature-gated backend branches drifting during file moves
- session/audio finalization behavior changing during extraction
- `clippy` failing due to either new warnings or missing local dependency cache
