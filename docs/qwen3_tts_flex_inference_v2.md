# Qwen3-TTS Talker Decode V2

## Summary

This document defines the Rust inference milestone after V1 prefill alignment.

Status: implemented in `tts_rs_qwen_burn` as a single-token `talker` decode step with Python reference coverage.

V2 goals:

- add `talker` single-step decode with KV cache support
- preserve the existing V1 prefill path and alignment behavior
- validate cache position, RoPE offset, and attention visibility against Python
- keep all model math inside Burn module/tensor/backend APIs
- extend the Python reference path for deterministic prefill-plus-decode cases

V2 non-goals:

- multi-step autoregressive sampling loop
- generated-mode `code_predictor` expansion
- waveform decoding through `speech_tokenizer`
- end-to-end `generate_custom_voice`
- text-side prompt assembly in Rust

The expected output of this stage is a numerically validated single-token decode path. It should be reusable by later stages that add multi-step generation and sampling.

## Scope

V2 covers the `talker` model cache path only:

- prefill initializes per-layer KV cache from a prepared prompt tensor
- decode consumes one prepared token embedding with `seq_len=1`
- decode uses the existing cache length as the RoPE/cache offset
- decode returns the new hidden state, logits, updated cache, and optional activations
- validation requires `batch=1`; public structs should not unnecessarily hard-code that limit

The V1 `forward_talker_prefill` behavior must remain stable. Adding cache support for decode must not change the existing deterministic prefill reference case.

## Rust Interfaces

Add a small decode-facing input/output surface without replacing the V1 prefill API.

Implemented input:

- `TalkerDecodeInput`
  - `inputs_embeds`: one token of shape `[batch, 1, hidden]`
  - `position_ids`: position ids for the current token, shape `[3, batch, 1]`
  - `attention_mask` (optional): full visible sequence mask when needed by the caller
  - `collect_activations`: debug flag

Implemented output:

- `TalkerDecodeOutput`
  - `last_hidden_state`
  - `logits`
  - `activations`

Implemented function:

- `forward_talker_decode_step(config, loaded, input, cache)`

Implementation constraints:

- use the existing `KeyValueCache<B>` and `AutoregressiveCache<B>` types
- compute cache offset from cache state, not from a duplicated caller-maintained counter
- keep Linear, MLP, RMSNorm, attention, RoPE, and tensor operations inside Burn APIs
- do not introduce backend- or dtype-specific helper math such as custom BF16/F32 linear accumulation
- keep checkpoint dtype semantics; tests may cast tensors to `float32` only for metrics

## Python Reference Alignment

`py/generate_reference.py` exports a deterministic V2 case into `reference.json`.

Reference behavior:

- load Python model with `dtype="auto"`
- run the same deterministic prepared prefill input as V1
- run one additional prepared decode token using Python `past_key_values` and `cache_position`
- export prefill logits, decode logits, decode hidden state, selected layer hidden outputs, and cache metadata needed for comparison

The Python reference is authoritative for cache semantics. Rust should match the Python sequence of:

1. prefill prompt embeddings
2. retain Python past key/value state
3. decode one token with the correct position/cache offset
4. compare the one-step decode output

Do not add a separate baseline comparison binary in V2. The alignment entrypoint remains an ignored Rust integration test that reads the Python reference artifact.

## Comparison Rules

Rust alignment compares:

- prefill output remains within V1 tolerance
- decode `last_hidden_state` shape and sample values
- decode logits shape, first values, max abs diff, and mean abs diff
- selected `layers.{i}.hidden.output` tensors for decode
- cache length before and after decode

Comparison policy:

- inference dtype follows checkpoint tensors
- summary statistics may be computed as `float32`
- full-logits max/mean absolute diff is more meaningful than logits sum alone for BF16 reductions
- any large drift must be localized to prefill, cache update, RoPE offset, mask visibility, or decode attention
- logits sums are diagnostics and intentionally looser than full-logits max/mean checks because BF16 reductions are backend-sensitive

## Test Plan

Required fast tests:

- cache length increments after decode
- decode rejects empty or multi-token inputs when the API requires a single step
- RoPE offset uses prior cache length
- attention mask/cache visibility permits attending to all cached tokens and the current token
- prefill path remains unchanged for the V1 reference case

Required validation commands:

1. `cargo test -p tts_rs_qwen_burn`
2. `uv run python py/generate_reference.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output reference.json`
3. `cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored --nocapture`
4. existing ignored roundtrip tests still pass after module or loader changes

Acceptance criteria:

- V1 prefill alignment still passes
- Rust decode cache length matches the expected Python step count
- Rust one-step decode logits and hidden activations are close enough for checkpoint-dtype Flex execution
- no dtype/backend-specific math helper is introduced in model code

## Next Stage

After V2 is stable, continue in this order:

1. add multi-step autoregressive token generation
2. extend code predictor from teacher-forced mode to generated mode
3. add sampling controls and stopping rules
4. implement `speech_tokenizer` decoder inference
5. connect generated talker codes to waveform decoding
6. decide whether to pull text-side preprocessing into Rust
