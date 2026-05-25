# Qwen3-TTS Talker Autoregressive Generation V3

## Summary

This document defines the third Rust inference milestone after V1 prefill alignment and V2 single-step decode alignment.

Status: design only. Do not implement Rust or Python code from this document until the phase is explicitly approved.

V3 goals:

- add deterministic multi-step autoregressive generation for the main `talker` model
- reuse the V1 prefill path and V2 single-step decode path
- keep KV cache, position ids, RoPE offset, and attention visibility driven by the cache state
- validate generated token ids and per-step logits against Python greedy generation
- keep all model math inside Burn module/tensor/backend APIs
- preserve checkpoint dtype semantics; tests may cast outputs to `float32` only for metrics

V3 non-goals:

- generated-mode `code_predictor` expansion
- sampling controls such as top-k, top-p, temperature, repetition penalty, or beam search
- waveform decoding through `speech_tokenizer`
- end-to-end `generate_custom_voice`
- Rust text-side prompt assembly
- custom backend- or dtype-specific math helpers

The expected output of this stage is a numerically validated greedy multi-token `talker` generation loop. It should become the reusable driver for later stages that add generated code predictor groups, sampling policy, and waveform decoding.

## Scope

V3 covers only the main `talker` autoregressive loop:

- run prefill once from prepared prompt embeddings
- choose the first generated token from prefill logits
- repeatedly embed the selected token and run one V2 decode step
- append exactly one generated token per loop iteration
- stop after `max_new_tokens`
- return generated token ids, selected logits diagnostics, and final cache state

Validation remains `batch=1`, matching the current Python reference flow. Public structs should still avoid unnecessary batch-size hard-coding so future backends and batched generation can be added later.

V3 must not change the V1 prefill numerics or the V2 single-step decode numerics.

## Rust Interfaces

Add a generation-facing API on top of existing prefill/decode functions. Do not replace `forward_talker_prefill` or `forward_talker_decode_step`.

Planned input:

- `TalkerGenerateInput`
  - `prefill_inputs_embeds`: prompt tensor with shape `[batch, prefill_len, hidden]`
  - `prefill_position_ids`: position ids with shape `[3, batch, prefill_len]`
  - `prefill_attention_mask` (optional): prompt visibility mask with shape `[batch, prefill_len]`
  - `max_new_tokens`: number of tokens to generate; must be greater than zero
  - `collect_step_diagnostics`: debug flag for alignment-only per-step data

Planned output:

- `TalkerGenerateOutput`
  - `generated_token_ids`: generated codec token ids with shape `[batch, max_new_tokens]`
  - `prefill_logits`: logits produced by the prefill call
  - `step_logits`: per-decode-step logits when diagnostics are enabled
  - `step_diagnostics`: optional per-step hidden/logit/cache metadata for alignment

Planned function:

- `generate_talker_tokens(config, loaded, input, cache)`

Generation rules:

1. validate cache layer count and reset/expect empty cache at generation start
2. run `forward_talker_prefill` with `collect_activations=false` by default
3. select token 0 from the last prefill position logits using greedy argmax
4. for each following step, build one-token `inputs_embeds` from `loaded.model.talker.model.codec_embedding`
5. use cache length as the current decode position for all three mRoPE position channels
6. build a full visible attention mask of length `cache_len + 1` when an attention mask is required
7. call `forward_talker_decode_step`
8. select the next token from the one-step decode logits using greedy argmax
9. append token id and continue until `max_new_tokens` tokens have been produced

Implementation constraints:

- use Burn tensor operations for argmax, slicing, embedding lookup, concatenation, and shape transforms
- do not copy tensor values to host except in tests or debug-only diagnostics
- do not introduce custom BF16/F32 accumulation, custom linear kernels, or backend-specific math paths
- use the checkpoint tensor dtype for inference
- keep logits comparison statistics in tests separate from model computation

## Python Reference Alignment

Extend `py/generate_reference.py` to export a deterministic V3 greedy generation case into `reference.json`.

Reference behavior:

- load Python model with `dtype="auto"`
- use the same deterministic prepared prefill case as V1/V2
- run greedy generation for a fixed small `max_new_tokens`, default `4`
- use Python `past_key_values` and `cache_position` for each decode step
- export enough data to validate token selection, cache progression, and logits drift

Reference artifact additions:

- `generation_input.max_new_tokens`
- `generation_expected.generated_token_ids`
- `generation_expected.prefill_selected_token_id`
- `generation_expected.steps.{i}.token_id`
- `generation_expected.steps.{i}.logits` with shape, first values, sum, and optional flattened values
- `generation_expected.steps.{i}.cache_len_before`
- `generation_expected.steps.{i}.cache_len_after`
- optional `generation_expected.steps.{i}.hidden.output` for drift localization

The Python reference is authoritative for generation semantics. Rust should match this sequence:

1. prefill prompt embeddings
2. select the first generated token from the last prefill logits
3. embed that token for decode step 1
4. decode with the current cache state
5. select the next token from decode logits
6. repeat until `max_new_tokens` tokens are produced

Do not add a separate baseline comparison binary in V3. The alignment entrypoint remains the ignored Rust integration test that reads the Python reference artifact.

## Comparison Rules

Rust alignment should compare:

- V1 prefill output remains within existing tolerance
- V2 first decode step remains within existing tolerance
- generated token id sequence exactly matches Python greedy output
- per-step cache length before and after decode matches Python
- per-step logits shape, first values, max abs diff, and mean abs diff
- selected hidden activations only when needed for drift localization

Comparison policy:

- generated token ids are exact-match assertions
- full-logits max/mean absolute diff is the primary numeric signal
- logits sums are diagnostics only because BF16 reductions are backend-sensitive
- any large drift must be localized to token selection, embedding lookup, cache update, RoPE position, attention mask visibility, or decode attention

## Test Plan

Required fast tests:

- generation rejects `max_new_tokens == 0`
- generation starts from an empty cache or explicitly resets it
- generated token tensor has shape `[batch, max_new_tokens]`
- cache length after generation equals `prefill_len + max_new_tokens - 1` after the final decode call
- greedy token selection uses the last prefill position for token 0
- decode position ids are derived from cache length
- no custom dtype/backend math helper is introduced

Required validation commands:

1. `cargo test -p tts_rs_qwen_burn`
2. `uv run python py/generate_reference.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output reference.json`
3. `cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored --nocapture`
4. existing ignored roundtrip tests still pass after module or loader changes

Acceptance criteria:

- V1 prefill alignment still passes
- V2 single-step decode alignment still passes
- Rust generated token ids exactly match Python for the deterministic greedy case
- per-step logits are close enough for checkpoint-dtype Flex execution
- cache lengths match Python at every generation step
- model code remains backend-portable Burn tensor/module code

## Next Stage

After V3 is stable, continue in this order:

1. extend `code_predictor` from teacher-forced mode to generated mode
2. add sampling controls and stopping rules
3. implement `speech_tokenizer` decoder inference
4. connect generated talker codes to waveform decoding
5. decide whether to pull text-side preprocessing into Rust
