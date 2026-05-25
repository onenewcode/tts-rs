# Qwen3-TTS Code Predictor Generation V4

## Summary

This document freezes the fourth Rust inference milestone after V1 prefill alignment, V2 single-step decode alignment, and V3 main `talker` autoregressive generation.

Status: implemented, validation-blocked. This document was persisted before Rust/Python V4 code changes.

V4 goals:

- add deterministic generated-mode `code_predictor` expansion for codec groups
- reuse the existing `talker` hidden state and generated base codec token from V3
- match Python code predictor generation semantics for greedy token selection
- validate generated codec group ids and per-step logits against Python
- keep all model math inside Burn module/tensor/backend APIs
- preserve checkpoint dtype semantics; tests may cast outputs to `float32` only for metrics

V4 non-goals:

- sampling controls such as top-k, top-p, temperature, repetition penalty, or beam search
- waveform decoding through `speech_tokenizer`
- end-to-end `generate_custom_voice`
- Rust text-side prompt assembly
- custom backend- or dtype-specific math helpers

The expected output of this stage is a numerically validated deterministic code-group expansion path. It should become the reusable bridge between V3 main-token generation and later waveform decoding.

## Rust Interfaces

Add generated-mode code predictor APIs alongside the existing teacher-forced entrypoint. Do not replace `forward_code_predictor_teacher_forced`.

Planned input:

- `CodePredictorGenerateInput`
  - `talker_hidden_state`: current main talker hidden state with shape `[batch, hidden]`
  - `base_codec_token_id`: main talker codec token with shape `[batch, 1]`
  - `collect_step_diagnostics`: debug flag for alignment-only per-step data

Planned output:

- `CodePredictorGenerateOutput`
  - `codec_ids`: complete codec groups with shape `[batch, num_code_groups]`
  - `predictor_token_ids`: generated predictor groups with shape `[batch, num_code_groups - 1]`
  - `step_logits`: per-group logits when diagnostics are enabled
  - `step_diagnostics`: per-step cache length metadata when diagnostics are enabled

Planned function:

- `generate_code_predictor_groups(config, loaded, input, cache)`

Generation rules:

1. validate code predictor cache layer count and reset it at generation start
2. embed `base_codec_token_id` with `talker.model.codec_embedding`
3. run code predictor prefill with `[talker_hidden_state, base_codec_embedding]`
4. select group 1 from the last prefill position using `lm_head[0]` greedy argmax
5. for following groups, embed the previously selected token with `code_predictor.model.codec_embedding[group_idx - 1]`
6. use code predictor cache length as the single-step decode position
7. apply `lm_head[group_idx]` to the one-step hidden state and greedily select the next group token
8. concatenate `base_codec_token_id` and all generated predictor tokens

Implementation constraints:

- use Burn tensor operations for argmax, slicing, embedding lookup, concatenation, and shape transforms
- do not copy tensor values to host except in tests or debug-only diagnostics
- do not introduce custom BF16/F32 accumulation, custom linear kernels, or backend-specific math paths
- use the checkpoint tensor dtype for inference
- keep logits comparison statistics in tests separate from model computation

## Python Reference Alignment

Extend `py/generate_reference.py` to export a deterministic V4 code predictor generation case into `reference.json`.

Reference behavior:

- load Python model with `dtype="auto"`
- use the V3 deterministic prefill case
- select a base codec token from the last main talker prefill logits
- manually expand `code_predictor` groups greedily using the official Python module semantics
- export enough data to validate token selection, cache progression, and logits drift

Reference artifact additions:

- `code_predictor_generation_input.base_codec_token_id`
- `code_predictor_generation_expected.codec_ids`
- `code_predictor_generation_expected.predictor_token_ids`
- `code_predictor_generation_expected.steps.{i}.token_id`
- `code_predictor_generation_expected.steps.{i}.logits`
- `code_predictor_generation_expected.steps.{i}.cache_len_before`
- `code_predictor_generation_expected.steps.{i}.cache_len_after`

The Python reference is authoritative for generated code predictor semantics.

## Test Plan

Required fast tests:

- code predictor generation rejects a wrong cache layer count
- generated `codec_ids` has shape `[batch, num_code_groups]`
- generated `codec_ids` first column equals `base_codec_token_id`
- predictor token count equals `num_code_groups - 1`
- final cache length equals `num_code_groups`
- first predictor token is selected from the prefill last position logits

Required validation commands:

1. `cargo test -p tts_rs_qwen_burn`
2. `uv run python py/generate_reference.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output reference.json`
3. `cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored --nocapture`
4. existing ignored roundtrip tests still pass after module or loader changes

Acceptance criteria:

- V1 prefill alignment still passes
- V2 single-step decode alignment still passes
- V3 main talker generation alignment still passes
- Rust generated codec group ids exactly match Python greedy code predictor generation
- per-step logits are close enough for checkpoint-dtype Flex execution
- cache lengths match Python at every code predictor generation step
- model code remains backend-portable Burn tensor/module code

Current validation note:

- Fast Rust tests pass.
- The alignment test now compares full flattened tensor values; it no longer accepts first/last edge-only checks or sum-based checks.
- Python reference generation is forced to eager attention so Rust and Python compare the same attention operator.
- Full alignment currently fails before V4 code predictor validation at `layers.1.attn_residual.output[4365]` with `diff=0.0078125` and fixed tolerance `0.005`. The first remaining source is a `0.00024414063` layer-0 MLP output drift that later crosses a bf16 residual rounding boundary.

## Stage Summary (V1-V4)

| Stage | Description | Rust Test | Python Reference |
|---|---|---|---|
| V1 | Prefill alignment | `tests/talker_alignment.rs` | `py/generate_reference.py` |
| V2 | Single-step decode alignment | `tests/talker_alignment.rs` | `py/generate_reference.py` |
| V3 | Talker autoregressive generation | `tests/talker_alignment.rs` | `py/generate_reference.py` |
| V4 | Code predictor generation | `tests/talker_alignment.rs` | `py/generate_reference.py` |

All four stages share the same Rust alignment test and Python reference generator.
The reference JSON (`reference.json`) contains prefill activations, decode step outputs,
generated token IDs, and code predictor outputs in a single file.

## Next Stage

After V4 is stable, continue in this order:

1. add sampling controls and stopping rules
2. implement `speech_tokenizer` decoder inference
3. connect generated talker codes to waveform decoding
4. decide whether to pull text-side preprocessing into Rust
