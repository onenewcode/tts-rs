# V9 Refactoring Proposal

V9 is a breaking refactor.

## Decisions

- Rename the old `speech_tokenizer` Rust module/API to `audio_codec`.
- Add `frontend` for Rust text tokenization, CustomVoice prompt construction, and
  prefill tensor construction.
- Keep production inference in Rust: tokenizer -> embeddings/prefill -> talker ->
  code predictor -> audio codec -> WAV.
- Use Python only as an alignment oracle under `py/generate_reference_v9_*.py`.
- Do not provide old API compatibility re-exports.

## Public API Shape

```rust
frontend::build_custom_voice_prefill_batch(...)
audio_codec::load_qwen3_tts_audio_codec(...)
talker::generate_talker_tokens(...)
talker::generate_code_predictor_groups(...)
shared::io::save_wav(...)
```

## Current First-Stage Status

Implemented:

- `audio_codec` module and public loader names.
- `frontend` tokenizer/prompt/prefill module.
- `shared::io::{save_wav, write_wav}`.
- `qwen3-tts` single-sample CLI.
- Talker generation returns per-token hidden states.
- Code predictor projection casts inputs to projection weight dtype.
- V9 tokenizer/prefill Python oracle command surfaces.

Remaining hardening:

- Full BF16 embedding numeric prefill oracle.
- Full ignored E2E oracle for talker tokens, codec groups, and waveform preview.
