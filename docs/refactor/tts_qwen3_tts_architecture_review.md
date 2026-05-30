# `tts_qwen3_tts` Architecture Review Document

## Summary

After comparing the current `tts_qwen3_tts` layout with mainstream inference frameworks and model-driver implementations, the current directory structure is **not reasonable as a long-term template**.

The main problem is not just depth. The deeper issue is that the crate is currently organized around an internal execution-component abstraction:

- `model/graph/engine/components/generator`
- `model/graph/engine/components/decoder`
- `spec`
- `lowering`

This makes the crate look like a small internal framework, while the intended role of `tts_qwen3_tts` is a **single model-family inference driver**.

For a driver crate intended to become a future reference template, the design should prioritize:

- simple inference-focused structure
- clear driver-layer boundaries
- model-private implementation grouped by real model sub-systems
- minimal internal abstraction
- no internal management/framework vocabulary

## Review of the Current Structure

Current top-level layout:

```text
src/
  backend/
  capabilities/
  execution/
  io/
  loading/
  model/
  profiling/
  runtime/
  surface/
```

Current `model/` layout:

```text
model/
  graph/
    engine/
      components/
        generator/
        decoder/
      spec.rs
  speaker.rs
  voice_clone.rs
```

### What is reasonable in the current structure

The following top-level responsibilities are directionally correct:

- `surface`
- `loading`
- `capabilities`
- `execution`

These match common driver-layer responsibilities found in larger inference ecosystems.

### What is not reasonable

#### 1. `backend/` as a crate-local layer

This is not model-family knowledge. It is runtime/backend policy.

Problems:

- every future driver crate would need the same layer
- backend selection is not Qwen3-specific
- it increases duplication and pushes management concerns into the driver

Conclusion:

- `backend/` should not exist as a dedicated layer inside `tts_qwen3_tts`

#### 2. `io/` as a dedicated layer

`io/` currently only holds tokenizer loading.

Problems:

- the boundary is fake
- tokenizer loading is compiler/frontend logic, not a reusable I/O subsystem
- the folder exists only because of one file

Conclusion:

- `io/` should be removed
- tokenizer should move into `execution/compiler/`

#### 3. `model/graph/engine/components/...`

This is the least reasonable part of the current architecture.

Problems:

- the main axis is execution-component vocabulary, not model-family structure
- the crate becomes harder to read because real Qwen3 sub-systems are hidden behind generic terms
- `speaker.rs` and `voice_clone.rs` already sit outside that structure, proving the structure does not match the real model
- `spec` and component-DAG abstractions are over-designed for a single inference driver

Conclusion:

- the entire `graph/engine/components/spec` structure should be removed
- the `model` layer should be rebuilt around real model sub-systems

#### 4. `voice_clone` currently sits at the wrong level

Current behavior shows that `voice_clone` is mostly request-time conditioning logic:

- validate transcript rules
- load reference audio
- compute speaker embedding
- compute reference codec frames
- assemble prompt

This is not a pure model sub-system.

Conclusion:

- `voice_clone` should move out of `model/`
- it belongs under `execution/`

#### 5. `speaker.rs` is too large and too mixed

Current `speaker.rs` mixes:

- config parsing
- weight loading
- mel feature extraction
- FFT/filterbank utilities
- network definition
- inference entrypoint

This is too much for one file.

Conclusion:

- `speaker` should be split into a dedicated subdirectory with explicit responsibilities

## Comparison with Mainstream Architectures

### Hugging Face Transformers

Transformers organizes by model family, then within a family separates concerns like:

- configuration
- modeling
- tokenization

It does not create deep execution-component trees for a single model family.

Implication for `tts_qwen3_tts`:

- organize around model-family sub-systems, not internal execution-component abstractions

Source:
- https://github.com/huggingface/transformers/blob/main/docs/source/en/models.md

### llama.cpp

When adding a new architecture, the focus is on:

- model architecture definition
- loading weights
- graph/build/runtime implementation

The implementation follows the model's real structure rather than introducing a deep generic component hierarchy.

Implication for `tts_qwen3_tts`:

- keep the model layer close to real Qwen3 sub-systems
- avoid introducing extra structure that exists only to describe structure

Source:
- https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md

### vLLM

vLLM clearly separates runtime orchestration from model implementation:

- loader logic
- execution/runtime logic
- model executor/model implementation

But a single model implementation does not become a deeply nested internal framework.

Implication for `tts_qwen3_tts`:

- `execution` and `model` should be separate
- `model` should stay simple and model-specific

Source:
- https://docs.vllm.ai/en/stable/api/vllm/model_executor/model_loader/

## Recommended Target Structure

### Top-level crate structure

```text
src/
  surface/
  loading/
  capabilities/
  execution/
  model/
```

### Remove entirely

```text
src/backend/
src/io/
src/model/graph/engine/
src/model/graph/engine/components/
src/model/graph/engine/spec.rs
```

### `execution/` structure

```text
src/execution/
  mod.rs
  compiler/
    mod.rs
    tokenizer.rs
    prompt.rs
    session_seed.rs
  conditioning.rs
  run.rs
  error.rs
  session.rs
```

Responsibilities:

- `compiler/tokenizer.rs`
  - tokenizer asset loading for compilation
- `compiler/prompt.rs`
  - prompt/token compilation helpers
- `compiler/session_seed.rs`
  - request-to-seed preparation
- `conditioning.rs`
  - voice-clone request conditioning
- `run.rs`
  - main single-request inference pipeline

### `model/` structure

```text
src/model/
  mod.rs
  common/
    mod.rs
    tensor.rs
    ops.rs
  talker/
    mod.rs
    config.rs
    weights.rs
    network.rs
    infer.rs
  codec/
    mod.rs
    config.rs
    weights.rs
    core.rs
    encode.rs
    decode.rs
  speaker/
    mod.rs
    config.rs
    weights.rs
    feature.rs
    network.rs
    infer.rs
```

## Rationale for the Model Structure

### `talker/`

Represents the acoustic generation model.

Use:

- `config.rs` for runtime configuration structures
- `weights.rs` for weight loading/remap
- `network.rs` for network definition
- `infer.rs` for generation primitives

### `codec/`

Represents the full audio codec subsystem.

This is important because the current code hides both encoder and decoder logic under a generic `decoder` concept, which is misleading.

Use:

- `config.rs` for codec configuration
- `weights.rs` for full codec assembly
- `core.rs` for shared codec internals
- `encode.rs` for reference-audio-to-codec-frame path
- `decode.rs` for codec-token-to-waveform path

### `speaker/`

Represents only the speaker embedding encoder.

It is not the whole voice-clone subsystem.

Use:

- `config.rs` for speaker encoder config
- `weights.rs` for speaker weight loading
- `feature.rs` for mel/FFT/filterbank preprocessing
- `network.rs` for speaker network definition
- `infer.rs` for `samples -> embedding`

### `common/`

Allowed only for genuinely shared math/tensor helpers.

This folder must stay strict.

Allowed:

- tensor helpers
- numerical ops
- reusable kernels

Not allowed:

- request logic
- loading logic
- conditioning logic
- backend policy

## Boundary Decisions

### `loading/`

Should own:

- package normalization
- manifest/model-dir parsing
- artifact path resolution
- raw config/asset discovery
- assembly of loaded runtime parts

Should not own:

- inference execution behavior
- prompt compilation
- conditioning flow

### `execution/`

Should own:

- request-time orchestration only

This includes:

- compiling prompts
- loading tokenizer frontend assets
- building voice clone conditioning
- invoking model primitives
- assembling final PCM result

### `model/`

Should own:

- model-private compute structures
- model-private runtime primitives
- loaded sub-systems
- shallow internal facade exposed to execution

Should not own:

- request orchestration
- prompt validation
- tokenizer/frontend concerns
- lifecycle/backend policy

## Final Assessment

### Is the current directory reasonable?

No.

It is too complex, too deep, and organized around the wrong axis. It overemphasizes internal component abstraction instead of expressing a simple inference driver for one model family.

### Is the proposed plan reasonable?

Yes.

The proposed plan is significantly more aligned with:

- mainstream model-driver organization
- your stated goal that each family crate should only do inference
- future use as a reference template

### Is there still room to optimize?

Yes, but mainly in tightening boundaries rather than changing the overall direction.

Remaining optimization points:

- keep `model/common` very strict
- keep `execution` with one clear main flow
- keep `model/mod.rs` shallow
- let `loading` parse assets, but not drift into runtime logic

## Recommended Final Direction

Adopt the following final architecture rule for `tts_qwen3_tts`:

- the crate remains an inference-only driver
- the top-level structure stays responsibility-based
- the model layer is rebuilt around real model sub-systems
- tokenizer moves under compiler
- voice clone moves under execution
- backend policy leaves the crate
- old `graph/engine/components/spec` structure is removed in one pass

## Assumptions

- `tts_qwen3_tts` is intended to become a template for future model-family drivers
- the template should emphasize clarity and maintainability over abstraction depth
- some local duplication is acceptable if it avoids reintroducing framework-like internal layers
