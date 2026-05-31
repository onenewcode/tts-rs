# Model Burn Tensor Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `tts_qwen3_tts/src/model` so common tensor helpers live in one place and internal tensor code follows Burn's recommended indexing, shape, and boundary patterns.

**Architecture:** Add a shared helper module under `src/model/nn`, migrate `talker` to it first, then fold `codec` and `speaker` into the same explicit boundary structure. Keep CPU DSP code intact, but concentrate tensor readbacks at named API boundaries.

**Tech Stack:** Rust, Cargo workspace tests, Burn tensors, existing `QwenTtsInferenceError` types

---

## File Map

- Create: `docs/superpowers/specs/2026-05-31-model-burn-tensor-refactor-design.md`
- Create: `docs/superpowers/plans/2026-05-31-model-burn-tensor-refactor.md`
- Create: `tts_qwen3_tts/src/model/nn/tensor.rs`
- Modify: `tts_qwen3_tts/src/model/nn/mod.rs`
- Modify: `tts_qwen3_tts/src/model/nn/sequence.rs`
- Modify: `tts_qwen3_tts/src/model/talker/infer.rs`
- Modify: `tts_qwen3_tts/src/model/talker/network.rs`
- Modify: `tts_qwen3_tts/src/model/codec/model.rs`
- Modify: `tts_qwen3_tts/src/model/codec/runtime/decode.rs`
- Modify: `tts_qwen3_tts/src/model/speaker/infer.rs`
- Modify: `tts_qwen3_tts/src/model/mod.rs`

### Task 1: Shared tensor helper module

**Files:**
- Create: `tts_qwen3_tts/src/model/nn/tensor.rs`
- Modify: `tts_qwen3_tts/src/model/nn/mod.rs`
- Modify: `tts_qwen3_tts/src/model/nn/sequence.rs`
- Test: `tts_qwen3_tts/src/model/nn/tensor.rs`

- [ ] **Step 1: Write the failing tests for shared helpers**

```rust
#[test]
fn flatten_hidden_3d_round_trips_shape() {
    let device = Default::default();
    let hidden = Tensor::<burn::backend::Flex, 3>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 2]),
        &device,
    );

    let flat = flatten_hidden_3d(hidden.clone());
    assert_eq!(flat.dims(), [2, 2]);

    let restored = unflatten_hidden_2d(flat, 1, 2);
    assert_eq!(restored.dims(), [1, 2, 2]);
}

#[test]
fn read_tensor_bool_reports_true_when_any_value_is_true() {
    let device = Default::default();
    let token = Tensor::<burn::backend::Flex, 2, burn::tensor::Int>::from_data(
        TensorData::new(vec![7_i64], [1, 1]),
        &device,
    );

    let flag = tensor_any_equal(token, 7).expect("readback should succeed");
    assert!(flag);
}
```

- [ ] **Step 2: Run the focused helper test target and verify it fails**

Run: `cargo test -p tts_qwen3_tts model::nn::tensor -- --nocapture`
Expected: FAIL because `tts_qwen3_tts::model::nn::tensor` and helper functions do not exist yet.

- [ ] **Step 3: Implement the minimal shared helper module**

```rust
// tts_qwen3_tts/src/model/nn/tensor.rs
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::error::QwenTtsInferenceError;

pub(crate) fn flatten_hidden_3d<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    hidden.reshape([batch_size * seq_len, hidden_size])
}

pub(crate) fn unflatten_hidden_2d<B: Backend>(hidden: Tensor<B, 2>, batch_size: usize, seq_len: usize) -> Tensor<B, 3> {
    let [_flat, hidden_size] = hidden.dims();
    hidden.reshape([batch_size, seq_len, hidden_size])
}

pub(crate) fn last_sequence_index<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![i32::try_from(seq_len - 1).unwrap()], [1]),
        device,
    )
}

pub(crate) fn tensor_any_equal<B: Backend>(tensor: Tensor<B, 2, Int>, value: usize) -> Result<bool, QwenTtsInferenceError> {
    let is_match = tensor
        .equal_elem(value as i64)
        .float()
        .reshape([1])
        .try_into_scalar()
        .map_err(|source| QwenTtsInferenceError::TensorRead {
            message: format!("tensor_any_equal: {source}"),
        })?
        .elem::<f32>();
    Ok(is_match > 0.5)
}
```

- [ ] **Step 4: Run the focused helper tests and verify they pass**

Run: `cargo test -p tts_qwen3_tts model::nn::tensor -- --nocapture`
Expected: PASS with the new helper tests green.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-31-model-burn-tensor-refactor-design.md \
        docs/superpowers/plans/2026-05-31-model-burn-tensor-refactor.md \
        tts_qwen3_tts/src/model/nn/tensor.rs \
        tts_qwen3_tts/src/model/nn/mod.rs \
        tts_qwen3_tts/src/model/nn/sequence.rs
git commit -m "model: add shared tensor helpers"
```

### Task 2: Move talker tensor flows onto shared helpers

**Files:**
- Modify: `tts_qwen3_tts/src/model/talker/infer.rs`
- Modify: `tts_qwen3_tts/src/model/talker/network.rs`
- Modify: `tts_qwen3_tts/src/model/talker/sampling.rs`
- Test: `tts_qwen3_tts/src/model/talker/sampling.rs`

- [ ] **Step 1: Write the failing regression tests for talker helper use**

```rust
#[test]
fn last_hidden_step_returns_rank_2_hidden_state() {
    let device = Default::default();
    let hidden = Tensor::<burn::backend::Flex, 3>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 2]),
        &device,
    );

    let last = last_hidden_step(hidden);
    assert_eq!(last.dims(), [1, 2]);
}
```

- [ ] **Step 2: Run the talker-focused test target and verify it fails for the expected reason**

Run: `cargo test -p tts_qwen3_tts talker:: -- --nocapture`
Expected: FAIL because the test references the new helper-backed behavior before the production code is updated.

- [ ] **Step 3: Refactor talker code to use shared helpers**

```rust
use crate::model::nn::tensor::{flatten_hidden_3d, tensor_any_equal, unflatten_hidden_2d};

fn selected_token_is_eos<B: Backend>(
    selected_token: &Tensor<B, 2, Int>,
    eos_token_id: Option<usize>,
) -> Result<bool, QwenTtsInferenceError> {
    let Some(id) = eos_token_id else {
        return Ok(false);
    };
    tensor_any_equal(selected_token.clone(), id)
}

let logits = self
    .codec_head
    .forward(flatten_hidden_3d(hidden_states.clone()))
        ;
let logits = unflatten_hidden_2d(logits, batch_size, seq_len);
```

- [ ] **Step 4: Run the talker tests and verify they pass**

Run: `cargo test -p tts_qwen3_tts talker:: -- --nocapture`
Expected: PASS with talker generation and sampling tests green.

- [ ] **Step 5: Commit**

```bash
git add tts_qwen3_tts/src/model/talker/infer.rs \
        tts_qwen3_tts/src/model/talker/network.rs \
        tts_qwen3_tts/src/model/talker/sampling.rs
git commit -m "talker: share burn tensor helpers"
```

### Task 3: Move codec tensor flows onto shared helpers

**Files:**
- Modify: `tts_qwen3_tts/src/model/codec/model.rs`
- Modify: `tts_qwen3_tts/src/model/codec/runtime/decode.rs`
- Test: `tts_qwen3_tts/src/model/codec/model.rs`

- [ ] **Step 1: Write the failing codec shape/readback regression tests**

```rust
#[test]
fn decoder_layer_keeps_batch_sequence_hidden_layout() {
    let device = Default::default();
    let hidden = Tensor::<burn::backend::Flex, 3>::zeros([1, 2, 8], &device);
    assert_eq!(hidden.dims(), [1, 2, 8]);
}
```

- [ ] **Step 2: Run the codec-focused tests and verify they fail for the expected reason**

Run: `cargo test -p tts_qwen3_tts codec:: -- --nocapture`
Expected: FAIL because codec helper extraction is not applied yet.

- [ ] **Step 3: Refactor codec flatten/unflatten and readback boundaries**

```rust
use crate::model::nn::tensor::{flatten_hidden_3d, unflatten_hidden_2d};

let hidden_2d = flatten_hidden_3d(hidden);
let projected = self.fc1.forward(hidden_2d);
let hidden = unflatten_hidden_2d(projected, batch_size, seq_len);
```

- [ ] **Step 4: Run codec tests and verify they pass**

Run: `cargo test -p tts_qwen3_tts codec:: -- --nocapture`
Expected: PASS with codec unit tests green.

- [ ] **Step 5: Commit**

```bash
git add tts_qwen3_tts/src/model/codec/model.rs \
        tts_qwen3_tts/src/model/codec/runtime/decode.rs
git commit -m "codec: consolidate tensor boundaries"
```

### Task 4: Normalize speaker and model boundary helpers

**Files:**
- Modify: `tts_qwen3_tts/src/model/speaker/infer.rs`
- Modify: `tts_qwen3_tts/src/model/mod.rs`
- Test: `tts_qwen3_tts/src/model/mod.rs`

- [ ] **Step 1: Write the failing boundary regression tests**

```rust
#[test]
fn flatten_reference_codec_frames_uses_quantizer_major_layout() {
    let frames = vec![vec![10, 20, 30], vec![11, 21, 31]];
    let flat = flatten_reference_codec_frames(&frames, 3).expect("frames should flatten");
    assert_eq!(flat, vec![10, 11, 20, 21, 30, 31]);
}
```

- [ ] **Step 2: Run the boundary-focused tests and verify they fail for the expected reason**

Run: `cargo test -p tts_qwen3_tts model:: -- --nocapture`
Expected: FAIL until the boundary helpers are updated to the shared conventions.

- [ ] **Step 3: Refactor speaker/model boundaries to the shared explicit readback style**

```rust
let embed = self.encoder.forward(mel.unsqueeze_dim::<3>(0).cast(self.encoder.dtype()));
read_float_tensor_1d(embed.reshape([self.encoder.enc_dim]), "speaker embedding")

let prefix = Tensor::<B, 3, Int>::from_data(
    TensorData::new(flat, [batch_size, num_quantizers, reference_codec_frames.len()]),
    device,
);
```

- [ ] **Step 4: Run the model tests and verify they pass**

Run: `cargo test -p tts_qwen3_tts model:: -- --nocapture`
Expected: PASS with shared-boundary tests green.

- [ ] **Step 5: Commit**

```bash
git add tts_qwen3_tts/src/model/speaker/infer.rs \
        tts_qwen3_tts/src/model/mod.rs
git commit -m "model: normalize tensor readback boundaries"
```

### Task 5: Full verification and Burn tensor audit

**Files:**
- Modify: `tts_qwen3_tts/src/model/**/*` as needed from audit findings
- Test: workspace subset for `tts_qwen3_tts`

- [ ] **Step 1: Run the focused `tts_qwen3_tts` tests**

```bash
cargo test -p tts_qwen3_tts -- --nocapture
```

- [ ] **Step 2: Search for remaining internal tensor readback sites**

```bash
rg -n "try_into_data|try_into_scalar|into_data\(|into_scalar\(" tts_qwen3_tts/src/model
```

Expected: only explicit output or boundary helpers remain, plus test-only reads.

- [ ] **Step 3: Fix any remaining direct internal readback or duplicated shape logic**

```rust
let logits = self.codec_head.forward(flatten_hidden_3d(hidden_states.clone()));
let logits = unflatten_hidden_2d(logits, batch_size, seq_len);

let is_eos = tensor_any_equal(selected_token.clone(), eos_token_id)
    .expect("boundary readback should succeed");
```

- [ ] **Step 4: Re-run verification after the audit fixups**

```bash
cargo test -p tts_qwen3_tts -- --nocapture
rg -n "try_into_data|try_into_scalar|into_data\(|into_scalar\(" tts_qwen3_tts/src/model
```

Expected: tests PASS and the remaining readback sites are intentional.

- [ ] **Step 5: Commit**

```bash
git add tts_qwen3_tts/src/model
git commit -m "model: align burn tensor flows"
```
