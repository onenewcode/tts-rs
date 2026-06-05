# Benchmarks

## Summary

The current benchmark is intentionally narrow: measure Qwen3-TTS custom-voice
`bf16` performance with a small hand-timed benchmark. `bf16` is the performance
standard for this repository.

Configuration is hard-coded in `tts_qwen3_tts/benches/common/mod.rs`:

- model: `Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice`
- text: `你好，欢迎使用 tts-rs。`
- language: `Chinese`
- speaker: `Vivian`
- sampling: greedy
- max new tokens: `32`
- warmup synthesis runs: `0`
- measured synthesis runs: `1`

## Run BF16 Benchmark

```bash
cargo bench -p tts_qwen3_tts --bench qwen3_custom_voice_bf16
```

The benchmark prints model loading and synthesis timing separately:

```text
qwen3 custom-voice bf16 benchmark
model_dir=...
text=你好，欢迎使用 tts-rs。
language=Chinese speaker=Vivian max_new_tokens=32
warmup_synthesis_runs=0 measured_synthesis_runs=1
load_ms=...
measure=1 synthesis_ms=... audio_s=... rtf=...
summary load_ms=... synthesis_ms=... audio_s=... rtf=...
```

Interpretation:

- `load_ms`: package resolution, weight loading, dtype conversion, and runtime
  assembly.
- `synthesis_ms`: synthesis only; model loading is excluded.
- `audio_s`: generated PCM duration.
- `rtf`: real-time factor, calculated as
  `synthesis_elapsed_seconds / generated_audio_seconds`.

## Notes

- The benchmark does not write WAV files; disk IO is excluded.
- The synthesis benchmark uses greedy sampling and a fixed token budget to reduce
  run-to-run noise.
- Keep benchmark changes outside `docs/TEST.md`; benchmark procedures live here.
