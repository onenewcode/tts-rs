# Qwen3-TTS Burn Stage 1

## Goal

Stage 1 only solves model definition and weight alignment:

1. Define Burn module trees for both Qwen3-TTS persisted checkpoints:
   - main `talker` checkpoint
   - `speech_tokenizer` checkpoint
2. Prove that every persisted tensor is mapped into the correct Burn path and roundtrips back exactly.

Inference is still out of scope.

## References

- Burn import example: `tracel-ai/burn/examples/import-model-weights`
- Burn structure reference: `tracel-ai/models/llama-burn`
- Local Python source:
  - `qwen_tts/core/models/modeling_qwen3_tts.py`
  - `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
  - `transformers/models/mimi/modeling_mimi.py`

## Local Assets

- Main config: `Qwen/*/config.json`
- Main checkpoint: `Qwen/*/model.safetensors`
- Speech tokenizer config: `Qwen/*/speech_tokenizer/config.json`
- Speech tokenizer checkpoint: `Qwen/*/speech_tokenizer/model.safetensors`
- Python baseline scripts: `py/`

## Rust Layout

The crate is now multi-file:

- `tts_rs_qwen_burn/src/lib.rs`
- `tts_rs_qwen_burn/src/talker.rs`
- `tts_rs_qwen_burn/src/speech_tokenizer.rs`
- `tts_rs_qwen_burn/src/manifest.rs`
- `tts_rs_qwen_burn/src/error.rs`
- `tts_rs_qwen_burn/src/paths.rs`

## Covered Modules

Main checkpoint:

- `talker.model`
- `talker.text_projection`
- `talker.codec_head`
- `talker.code_predictor`

Speech tokenizer checkpoint:

- `decoder.pre_transformer`
- `decoder.quantizer`
- `decoder.pre_conv`
- `decoder.upsample`
- `decoder.decoder`
- `encoder.encoder`
- `encoder.encoder_transformer`
- `encoder.downsample`
- `encoder.quantizer`

## Architecture Notes

`talker` is already modeled as a direct Burn module tree.

`speech_tokenizer` is now also expressed with explicit submodule types instead of generic weight slots:

- decoder wave stack:
  - input conv
  - 4 upsample stages
  - output `SnakeBeta`
  - output conv
- each decoder upsample stage is modeled as:
  - pre-activation `SnakeBeta`
  - causal transposed conv
  - 3 explicit residual units
- each decoder residual unit is modeled as:
  - `act1`
  - `conv1`
  - `act2`
  - `conv2`
- encoder backbone is modeled as:
  - input conv
  - 4 alternating resnet/downsample stages
  - output conv
- encoder backbone sparse indices are preserved with enum variants so checkpoint paths remain byte-for-byte aligned.
- encoder `_frame_rate` semantics are now preserved explicitly when deriving the final downsample layer.

## Burn Reuse

Burn native modules reused directly where possible:

- `Embedding`
- `Linear`
- `Conv1d`
- `ConvTranspose1d`
- `LayerNorm`
- `RmsNorm`

Custom `Param<Tensor<...>>` holders are used only where the Python models persist raw parameters or buffers:

- `SnakeBeta.{alpha,beta}`
- `LayerScale.scale`
- decoder codebooks `cluster_usage` / `embedding_sum`
- encoder codebooks `initialized` / `cluster_usage` / `embed_sum`

## Mapping Rules

Important facts:

- PyTorch linear weights are `[out, in]`; Burn `Linear` needs `PyTorchToBurnAdapter`.
- Burn `LayerNorm` names are handled by Burn's adapter automatically.
- Burn `RmsNorm` still needs explicit key remapping.

Load remappers:

- Talker:
  - `"(.*)norm\\.weight$" -> "${1}norm.gamma"`
- Speech tokenizer decoder RMSNorm only:
  - `"^(decoder\\.pre_transformer(?:\\.layers\\.\\d+\\.(?:input_layernorm|post_attention_layernorm)|\\.norm))\\.weight$" -> "${1}.gamma"`

Export remappers reverse the same RMSNorm name changes.

## Verification Strategy

The authoritative check is not `load_from` alone. Stage 1 uses full manifest parity:

1. Load the checkpoint into Burn.
2. Export the Burn model back into a PyTorch-view tensor set.
3. Build a deterministic JSON manifest for both source and exported tensors:
   - `path`
   - `shape`
   - `dtype`
   - `sha256`
4. Compare the full manifest exactly.

If key set, shape, dtype, and sha256 all match, the Rust structure and weight mapping are correct.

## Artifact Files

Rust default outputs:

- `artifacts/qwen3_tts/talker/source_manifest.json`
- `artifacts/qwen3_tts/talker/rust_export_manifest.json`
- `artifacts/qwen3_tts/talker/comparison_report.json`
- `artifacts/qwen3_tts/speech_tokenizer/source_manifest.json`
- `artifacts/qwen3_tts/speech_tokenizer/rust_export_manifest.json`
- `artifacts/qwen3_tts/speech_tokenizer/comparison_report.json`

Python outputs:

- `artifacts/qwen3_tts/talker/python_source_manifest.json`
- `artifacts/qwen3_tts/talker/python_structure_report.json`
- `artifacts/qwen3_tts/talker/python_vs_rust_report.json`
- `artifacts/qwen3_tts/speech_tokenizer/python_source_manifest.json`
- `artifacts/qwen3_tts/speech_tokenizer/python_vs_rust_report.json`

## Python Baselines

- `py/dump_talker_keys.py`
  - writes the canonical talker source manifest
- `py/verify_talker_structure.py`
  - writes a config-vs-checkpoint structure report
- `py/dump_speech_tokenizer_keys.py`
  - writes the canonical speech tokenizer source manifest
- `py/compare_manifests.py`
  - compares any two manifest files exactly and writes a JSON report

## Current Validation Targets

- Talker:
  - 402 tensors exact-match
- Speech tokenizer:
  - 496 tensors exact-match

For `speech_tokenizer`, Burn's load report can still show `unused=36`, but the exported manifest matches the source manifest exactly. The manifest parity is the stronger check and is the acceptance criterion for this stage.

## Next Stages

1. Add Burn-side forwards for the loaded structures.
2. Add numerical parity checks on selected submodules.
3. Start non-streaming end-to-end inference.
