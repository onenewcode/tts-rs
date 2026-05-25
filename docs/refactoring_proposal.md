# Refactoring Proposal: Domain-Module Architecture

## Current vs Target

| Dimension | Current | Target |
|---|---|---|
| Top-level | `talker/` + `speech_tokenizer/` (model names) | `talker/` + `tokenizer/` + `shared/` (domain + shared) |
| Constructor | `init/` directory, separate from model | Factory functions in domain |
| Sampling | Buried in `talker/inference.rs` | `shared/runtime/sampling.rs` |
| Config | Two config files in separate dirs | `shared/config/` |
| nn/ | `talker/nn/` only | `shared/nn/` (shared) + domain extensions |
| Loading | Two load files + two remap files | `shared/io/load.rs` |
| Test | In-domain + integration | Same pattern, cleaned up |

## Migration Phases

### Phase 2: Create `shared/` (Mechanical Moves — No Logic Changes)

**Goal**: Extract all shared infrastructure into `shared/`.

**Moves**:
- `talker/config.rs` + `speech_tokenizer/config.rs` → `shared/config/talker.rs` + `shared/config/tokenizer.rs`
- `talker/nn/{attention,layer,mlp,rms_norm}.rs` → `shared/nn/{attention,layer,mlp,rms_norm}.rs`
- `speech_tokenizer/model/common.rs` → `shared/nn/{conv,activation}.rs`
- `talker/cache.rs` → `shared/runtime/cache.rs`
- Extract `sample_token()` + `SamplingConfig` from `talker/inference.rs` → `shared/runtime/sampling.rs`
- `talker/load.rs` + `speech_tokenizer/load.rs` → `shared/io/load.rs`
- `talker/remap.rs` + `speech_tokenizer/remap.rs` → merge into `shared/io/load.rs`
- `talker/verify.rs` + `speech_tokenizer/verify.rs` → `shared/verify/weights.rs`
- `src/{error,manifest,paths}.rs` → `shared/{error,manifest,paths}.rs`

**Risk**: Low. Pure file moves + import path updates. No logic changes.

**Verification**: `cargo test -p tts_rs_qwen_burn` — all existing tests pass.

### Phase 3: Create `talker/` and `tokenizer/` (Domain Extraction)

**Goal**: Reorganize remaining code into domain modules.

**Moves**:
- `talker/model.rs` + `talker/init.rs` → `talker/model.rs` + `talker/factory.rs`
- `talker/inference.rs` → `talker/inference.rs` + `talker/types.rs`
- `talker/nn/rope.rs` → `talker/nn/rope.rs` (keep, talker-specific)
- `speech_tokenizer/model/decoder.rs` → `tokenizer/model.rs` + `tokenizer/transformer.rs` + `tokenizer/wave.rs`
- `speech_tokenizer/init/` → `tokenizer/factory.rs`
- `speech_tokenizer/inference.rs` → `tokenizer/inference.rs` + `tokenizer/types.rs`

**Risk**: Medium. Splitting large files and extracting factory logic.

**Verification**: `cargo test -p tts_rs_qwen_burn` — all existing tests pass.

### Phase 4: Cleanup

**Goal**: Delete old directories, update public API, finalize.

- Delete old `src/talker/` (replaced by new `talker/`)
- Delete old `src/speech_tokenizer/` (replaced by new `tokenizer/`)
- Update `lib.rs` public API
- Update binary imports
- Add module-level doc comments
- Move tests to `tests/`

**Risk**: Low. Cosmetic cleanup.

**Verification**: Full test suite + `cargo build --bin e2e`.

## Rollback Strategy

Each phase is a git commit. If any phase breaks, revert that commit.
Phases are independent — Phase 2 can ship without Phase 3.

## Current State (After Phase 1)

Phase 1 adds only documentation. No code moves. See `docs/architecture.md` for
the target state.
