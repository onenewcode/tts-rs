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
cargo test -p tts_rs_qwen_burn audio_codec::
```

These tests do not require local model weights.

## Slow Roundtrip Tests

The real-checkpoint tests are marked `ignored` because they are slow and require local assets.

Run them explicitly:

```bash
cargo test -p tts_rs_qwen_burn --test talker_roundtrip -- --ignored --nocapture
cargo test -p tts_rs_qwen_burn --test audio_codec_roundtrip -- --ignored --nocapture
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
- `artifacts/qwen3_tts/audio_codec/test_roundtrip/`

Each directory contains:

- `source_manifest.json`
- `rust_export_manifest.json`
- `comparison_report.json`

## Failure Notes

- If the test cannot find a model directory, set `QWEN_TTS_MODEL_DIR` explicitly.
- If roundtrip verification fails, inspect `comparison_report.json` first. That will tell you whether the regression is in key sets, shapes, dtypes, or tensor bytes.
- The talker slow test is expected to take much longer than the default unit suite.

## Talker Python Alignment

The talker inference path is aligned against a deterministic Python reference consumed by `tts_rs_qwen_burn/tests/talker_alignment.rs`. This is the current required Python-vs-Rust validation path; no separate baseline comparison binary is required.

Generate or refresh the reference file:

```bash
uv run python py/generate_reference.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output reference.json
```

This writes `reference.json` at the workspace root. The file contains checkpoint-dtype prefill inputs, a deterministic one-token decode input, per-layer hidden activation stats, final norm stats, cache lengths, and logits stats/values. The Python model is loaded with `dtype="auto"`, so it follows the checkpoint tensor dtype.

Run the Rust alignment test:

```bash
cargo test -p tts_rs_qwen_burn --test talker_alignment -- --ignored --nocapture
```

Notes:

- The Rust path keeps checkpoint tensor dtypes for Flex execution; tests may cast outputs to `float32` only when computing comparison statistics.
- Model code should use Burn modules and tensor APIs directly. Do not introduce backend- or dtype-specific math helpers to force Python-like accumulation.
- The Python exporter may print a SoX warning during import; that does not block reference generation.
- Alignment compares prefill and one-step decode shapes, cache length before/after decode, selected activation stats, and full-logits max/mean absolute differences.
- Logits sums are printed as diagnostics, but full-logits max/mean absolute diff is the primary numeric signal for checkpoint-dtype Flex runs.
