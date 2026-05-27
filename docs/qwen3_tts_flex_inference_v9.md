# Qwen3-TTS Flex Inference V9

V9 moves the remaining production path into Rust and makes Python a test oracle
only.

## Scope

- Local 12Hz `0.6B CustomVoice`.
- No streaming, voice cloning, or speaker encoder.
- Library API supports batches in the frontend; CLI supports one sample.

## Rust Pipeline

1. `Qwen3TtsTextTokenizer` loads `vocab.json`, `merges.txt`, and
   `tokenizer_config.json`.
2. `frontend::build_custom_voice_prefill_batch()` builds the fixed CustomVoice
   prompt:

   ```
   <|im_start|>assistant
   {text}<|im_end|>
   <|im_start|>assistant
   ```

3. Text ids pass through `talker.model.text_embedding` and
   `talker.text_projection`.
4. `generate_talker_tokens()` returns generated token ids and matching hidden
   states.
5. Each generated token/hidden-state pair feeds
   `generate_code_predictor_groups()`.
6. Codec groups are stacked as `[batch, num_code_groups, time_steps]` and decoded
   by `audio_codec::decode_codec_tokens()`.
7. `shared::io::save_wav()` writes `0000.wav`.

## CLI

```sh
cargo run --release -p tts_cli --bin tts_cli -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir output \
  --max-new-tokens 256
```

## Python Oracles

- `py/generate_reference_v9_tokenizer.py`
- `py/generate_reference_v9_prefill.py`

Generated JSON artifacts are written to `target/tmp` and are not committed.

## Decoder Parity Notes

The audio codec decoder follows the working Rust reference ordering:
`quantizer -> pre_conv -> pre_transformer -> upsample -> wave decoder`.
ConvNeXt uses GELU, vocoder kernel-7 convs use causal left padding, wave
transposed convs trim `kernel_size - stride` samples from both sides, SnakeBeta
uses exponentiated alpha/beta in F32, and the final waveform is clamped to
`[-1, 1]`.
