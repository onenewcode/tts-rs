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
- build a Python baseline export path
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
Instead, Rust forward entrypoints will receive already prepared tensors from the baseline case.

## Rust Interfaces

The public Rust inference surface is kept small and deterministic, following the `Module-centric` design and `llama-burn` best practices.

Key Architectural Patterns:

- **Module-centric Forward**: Every component (e.g., `Qwen3TtsAttention`, `Qwen3TtsDecoderLayer`) implements a standard `forward` method. Higher-level modules orchestrate the flow by passing data and state (like cache and RoPE) downward.
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
- compute in `float32` on Flex
- KV cache support enabled for future-proofing architecture

## Python Baseline

Baseline exporter:

- `py/export_talker_baseline.py`

Exporter behavior:
- load the local Qwen3-TTS checkpoint on CPU
- force `eval()`
- run small deterministic cases
- export inputs, key activations, and outputs

Baseline artifact layout:

- `artifacts/qwen3_tts/talker/baseline/<case>/case.json`
- `artifacts/qwen3_tts/talker/baseline/<case>/inputs.safetensors`
- `artifacts/qwen3_tts/talker/baseline/<case>/activations.safetensors`
- `artifacts/qwen3_tts/talker/baseline/<case>/outputs.safetensors`

Key activation list:

- `text_projection.output`
- `codec_embedding.output`
- `layers.{i}.attn.output`
- `layers.{i}.mlp.output`
- `layers.{i}.hidden.output`
- `model.norm.output`
- `codec_head.logits`
- `code_predictor.input_embeds`
- `code_predictor.layers.{i}.hidden.output`
- `code_predictor.logits`

V1 intentionally does not export:

- q/k/v split tensors
- rotary-pre / rotary-post tensors
- attention probability tensors

## Comparison Rules

Rust baseline runner should:

- load the exported case metadata
- load tensor payloads
- run Rust Flex forward
- compare expected tensors by name
- emit a JSON report

Report path:

- `artifacts/qwen3_tts/talker/rust_vs_python/<case>/comparison_report.json`

Comparison defaults:

- dtype normalized to `float32`
- `atol = 1e-3`
- `rtol = 1e-3`
- report shape match, max abs diff, mean abs diff, pass/fail

## Test Plan

Required fast tests:

- RMSNorm behavior
- rotary helper shape logic
- repeat-kv behavior
- causal mask creation
- attention output shapes
- decoder layer residual behavior
- code predictor teacher-forced input assembly

Required baseline cases:

1. `prefill_small_seq`
2. `subtalker_teacher_forced`

Required validation commands:

1. `cargo test -p tts_rs_qwen_burn`
2. exporter command for Python baseline generation
3. Rust baseline comparison command
4. existing ignored roundtrip tests still pass

## Next Stage

After V1 is numerically stable, continue in this order:

1. add `talker` single-step decode with cache support
2. add multi-step autoregressive token generation
3. extend code predictor from teacher-forced mode to generated mode
4. implement `speech_tokenizer` decoder inference
5. connect `talker` output codes to waveform decoding
6. decide whether to pull text-side preprocessing into Rust

The V1 artifact format and comparison report should remain reusable in later stages.
able in later stages.
