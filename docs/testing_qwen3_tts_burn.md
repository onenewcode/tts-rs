# Testing Qwen3 TTS Burn

## Test Layers

The crate now separates tests into two layers:

- Fast unit tests: structure assembly, remapper rules, manifest comparison, filesystem helpers.
- Slow roundtrip integration tests: load real local Qwen weights and verify manifest parity end to end.

Use fast tests during normal development. Run slow tests after changing module trees, remappers, or checkpoint loading.

## Fast Tests

Run the default suite:

```bash
cargo test -p tts_rs_qwen_burn
```

Useful focused runs:

```bash
cargo test -p tts_rs_qwen_burn manifest::
cargo test -p tts_rs_qwen_burn paths::
cargo test -p tts_rs_qwen_burn talker::
cargo test -p tts_rs_qwen_burn speech_tokenizer::
```

These tests do not require local model weights.

## Slow Roundtrip Tests

The real-checkpoint tests are marked `ignored` because they are slow and require local assets.

Run them explicitly:

```bash
cargo test -p tts_rs_qwen_burn --test talker_roundtrip -- --ignored --nocapture
cargo test -p tts_rs_qwen_burn --test speech_tokenizer_roundtrip -- --ignored --nocapture
```

Model discovery works in this order:

1. If `QWEN_TTS_MODEL_DIR` is set, the tests use that directory.
2. Otherwise the crate scans `Qwen/*` under the workspace root and picks the first directory that contains both `config.json` and `model.safetensors`.

Example:

```bash
QWEN_TTS_MODEL_DIR=/path/to/Qwen/Qwen3-TTS cargo test -p tts_rs_qwen_burn --test talker_roundtrip -- --ignored --nocapture
```

## Expected Outputs

Slow tests write verification artifacts into:

- `artifacts/qwen3_tts/talker/test_roundtrip/`
- `artifacts/qwen3_tts/speech_tokenizer/test_roundtrip/`

Each directory contains:

- `source_manifest.json`
- `rust_export_manifest.json`
- `comparison_report.json`

## Failure Notes

- If the test cannot find a model directory, set `QWEN_TTS_MODEL_DIR` explicitly.
- If roundtrip verification fails, inspect `comparison_report.json` first. That will tell you whether the regression is in key sets, shapes, dtypes, or tensor bytes.
- The talker slow test is expected to take much longer than the default unit suite.

## Python Baseline Comparison

The talker inference path now has a separate Python-vs-Rust baseline flow.

Export deterministic Python baseline cases:

```bash
python py/export_talker_baseline.py --case all
```

This writes:

- `artifacts/qwen3_tts/talker/baseline/prefill_small_seq/`
- `artifacts/qwen3_tts/talker/baseline/subtalker_teacher_forced/`

Run Rust comparison against those cases:

```bash
cargo run -p tts_rs_qwen_burn --bin compare_qwen3_tts_talker_baseline -- --case-dir artifacts/qwen3_tts/talker/baseline/prefill_small_seq
cargo run -p tts_rs_qwen_burn --bin compare_qwen3_tts_talker_baseline -- --case-dir artifacts/qwen3_tts/talker/baseline/subtalker_teacher_forced
```

Comparison reports are written to:

- `artifacts/qwen3_tts/talker/rust_vs_python/prefill_small_seq/comparison_report.json`
- `artifacts/qwen3_tts/talker/rust_vs_python/subtalker_teacher_forced/comparison_report.json`

Notes:

- The compare binary uses the inference-specific talker loader, which casts half-precision checkpoint weights to `float32` for Flex execution.
- The Python exporter only needs the local model directory. It may print a SoX warning during import; that does not block baseline export.
- Current baseline tolerance is `atol=1e-3`, `rtol=1e-3`, stored in each case's `case.json`.
