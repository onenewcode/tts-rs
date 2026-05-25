# Qwen3-TTS Flex Inference V1

## Summary

This document freezes the implementation plan for the first Rust inference milestone.

V1 goals:

- implement `talker` forward inference on `burn::backend::Flex`
- adopt `Module-centric` architecture with `forward` methods on all layers
- use official `burn::nn::attention` and `burn::nn::RotaryEncoding`
- leverage `burn::nn::transformer` utilities to reduce boilerplate
- implement modular `KV cache` support in the forward pass
- keep weights loaded from the existing `tts_rs_qwen_burn` structures
- build a Python reference export path for deterministic alignment
- compare key intermediate activations and final logits against Python

V1 non-goals:

- waveform decoding through `speech_tokenizer`
- end-to-end `generate_custom_voice`
- streaming inference
- sampling loop

Primary references:

- Burn Attention: <https://burn.dev/docs/burn/nn/modules/attention/index.html>
- Burn RotaryEncoding: <https://burn.dev/docs/burn/nn/modules/struct.RotaryEncoding.html>
- Burn Transformer: <https://burn.dev/docs/burn/nn/modules/transformer/index.html>
- llama-burn: <https://github.com/tracel-ai/models/tree/main/llama-burn/src>

## Scope

V1 only covers the `talker` stack:

- `Qwen3TtsTalkerResizeMlp`
- `Qwen3TtsTalkerModel`
- `Qwen3TtsAttention` (wrapping/extending official attention)
- `Qwen3TtsTextMlp`
- `Qwen3TtsDecoderLayer`
- `Qwen3TtsTalkerCodePredictorForConditionalGeneration` in teacher-forced mode

V1 will not attempt to reconstruct the full Python text-side prompt assembly logic.
Instead, Rust forward entrypoints receive already prepared tensors from the Python reference case.

## Rust Interfaces

The public Rust inference surface is kept small and deterministic, following the `Module-centric` design and `llama-burn` best practices.

Key Architectural Patterns:

- **Module-centric Forward**: Every component (e.g., `Qwen3TtsAttention`, `Qwen3TtsDecoderLayer`) implements a standard `forward` method. Higher-level modules orchestrate the flow by passing data and state (like cache and RoPE) downward.
- **Burn backend abstraction**: Linear layers, MLPs, heads, attention, and tensor math must use Burn modules/tensors so the model remains portable across backends. Do not add dtype-specific helper paths such as custom BF16/F32 linear accumulation.
- **Autoregressive Cache**: We use an `AutoregressiveCache` (ref. `llama-burn/src/cache.rs`) that manages pre-allocated tensor slices and sliding window logic. `KeyValueCache` is a standard container for these caches per layer.
- **Official RoPE Integration**: We use `burn::nn::RotaryEncoding` for standard RoPE (e.g., in the CodePredictor). For the multimodal `mRoPE` in the main Talker, we follow the same `apply(x, offset)` interface but use interleaved modality frequencies as required by Qwen3.

Planned inputs:

- `TalkerForwardInput`
  - `inputs_embeds`
  - `position_ids`
  - `attention_mask` (optional)
  - `collect_activations` (debug flag)
- `CodePredictorTeacherForcedInput`
  - `talker_hidden_states`
  - `codec_ids`
  - `position_ids` (optional)
  - `attention_mask` (optional)
  - `collect_activations` (debug flag)

Inference functions (e.g., `forward_talker_prefill`) now take an additional `&mut [KeyValueCache<B>]` argument, allowing for modular state management during both prefill and incremental generation.


V1 execution rules:

- batch-first tensors
- validation path only requires `batch=1`
- compute on Flex using the checkpoint tensor dtype; comparisons may cast outputs to `float32` only for statistics
- KV cache support enabled for future-proofing architecture

## Python Reference Alignment

Reference exporter:

- `py/generate_reference.py`

Exporter behavior:

- load the local Qwen3-TTS talker checkpoint on CPU
- use `dtype="auto"` so Python follows the checkpoint tensor dtype
- force `eval()`
- run a deterministic small prefill case
- export prepared inputs, selected layer activations, final norm, and logits to `reference.json`

Reference artifact:

- `reference.json`

Reference contents:

- `input.inputs_embeds`
- `input.position_ids`
- `expected.layers.{i}.hidden.output`
- `expected.layer_0_output` compatibility alias
- `expected.final_norm`
- `expected.logits` with shape, sum, first values, and flattened values

V1 intentionally does not require a separate baseline comparison binary. The current validation path is the ignored Rust integration test `tts_rs_qwen_burn/tests/talker_alignment.rs`, which consumes `reference.json` directly.

## Comparison Rules

Rust alignment should:

- load `reference.json`
- load the local checkpoint through the inference loader
- run Rust Flex talker prefill with `collect_activations=true`
- compare expected tensors by name
- print the first layer or output where drift becomes visible

Comparison defaults:

- inference dtype follows checkpoint tensors
- model math stays inside Burn tensor/module/backend APIs
- tests may cast outputs to `float32` only for statistics
- compare shape, sum, first values, logits max abs diff, and logits mean abs diff
- logits full-tensor max/mean diff is more meaningful than sum alone for BF16 reductions

## Test Plan

Required fast tests:

- RMSNorm behavior
- rotary helper shape logic
- repeat-kv behavior
- causal mask creation
- attention output shapes
- decoder layer residual behavior
- code predictor teacher-forced input assembly

Required alignment case:

1. deterministic talker prefill from `reference.json`

Required validation commands:

1. `cargo test -p tts_rs_qwen_burn`
2. `uv run python py/generate_reference.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output reference.json`
3. `cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored --nocapture`
4. existing ignored roundtrip tests still pass

## Next Stage

After V1 is numerically stable, continue in this order:

1. add `talker` single-step decode with cache support
2. add multi-step autoregressive token generation
3. extend code predictor from teacher-forced mode to generated mode
4. implement `speech_tokenizer` decoder inference
5. connect `talker` output codes to waveform decoding
6. decide whether to pull text-side preprocessing into Rust

The V1 reference format should remain reusable in later stages.
