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
