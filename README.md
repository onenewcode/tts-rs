# tts-rs

Rust workspace for running Qwen3-TTS locally and writing WAV output from the
CLI. The Qwen3-TTS runtime path is implemented on Burn.

Workspace crates:

- `tts_core` (`tts_infer/`): framework core for lifecycle, session/audio primitives, and loaded-instance management
- `tts_error`: shared diagnostics foundation
- `tts_qwen3_tts`: Qwen3-TTS driver crate
- `tts_app`: application-service orchestration for local frontends
- `tts_cli`: command-line shell that writes `.wav` files through `tts_app`

The CLI writes mono, 24 kHz, 16-bit PCM WAV output. Base voice cloning supports
both `ref_audio + ref_text` ICL conditioning and `x_vector_only` conditioning.

Important: run all CLI synthesis commands in release mode. Debug builds are
much slower and can look like they are hanging during model load or generation.

## Prerequisites

- Rust toolchain with Cargo
- A local Qwen3-TTS model directory downloaded from Hugging Face or copied from
  another machine

## Quickstart

The normal path is: pass the model directory directly.

Base voice clone ICL example:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone ICL smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --ref-text "Hello from the generated reference clip." \
  --backend flex \
  --output ./out/base_clone_icl_release.wav
```

Base x-vector-only example:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone x-vector-only smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --x-vector-only \
  --backend flex \
  --output ./out/base_clone_xvector_release.wav
```

Custom voice example:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --backend flex \
  --output ./out/custom-voice.wav
```

Custom voice instruct example:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --backend flex \
  --output ./out/custom-voice-instruct.wav
```

Notes:

- `--model-dir` should point at the model folder itself, not its parent
- `./out/` is created automatically if it does not exist
- `--speaker` only applies to `synthesize custom-voice`
- `--instruct` only applies to `synthesize custom-voice`; it describes the
  speaking style for the target text
- `--ref-text` is the transcript of `--ref-audio`, not the target text to
  synthesize
- `--x-vector-only` uses only the speaker embedding from `--ref-audio` and does
  not accept `--ref-text`
- language names are matched case-insensitively against the model metadata, so
  `Chinese` and `chinese` both work
- `--backend flex` is the recommended local default in this repository
- `--max-new-tokens` is optional; when omitted, the CLI uses the model package
  `generation_config.max_new_tokens` instead of applying a hard-coded CLI cap

## Expected Model Layout

By default the CLI expects the same file layout that the Hub model directory
already provides:

```text
<model-dir>/
  config.json
  generation_config.json
  model.safetensors
  tokenizer.json        # optional if vocab.json + merges.txt exist
  vocab.json
  merges.txt
  speech_tokenizer/
    config.json
    model.safetensors
```

Current repo-local examples:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

The CLI reads runtime control metadata directly from the model `config.json`.
You do not need to prepare a separate `control_config.json`.

## Supported Languages And Speakers

Language names come from the model metadata in `config.json`.

For the checked-in Qwen3 models in this repository, common values include:

- `Chinese`
- `English`
- `Japanese`
- `Korean`
- `German`
- `French`
- `Russian`
- `Portuguese`
- `Spanish`
- `Italian`

The custom-voice checkpoint in `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
contains speakers such as:

- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ryan`
- `Aiden`
- `Ono_Anna`
- `Sohee`

If you pass an unsupported language or speaker, the CLI reports the values that
the model actually supports.

## Advanced: Custom Manifest

Most users should not use this mode.

`--manifest` exists for non-standard layouts where your files are not stored in
the default Hub directory structure. In that case, point the CLI at a
`qwen3_tts_package.yaml` file.

This YAML only points to files that already exist in the model package. You do
not create any extra JSON config just for `tts-rs`.

Example:

```yaml
format: qwen3_tts_package/v1
name: custom-qwen3-layout

artifacts:
  tokenizer: ./tokenizer/vocab.json
  talker_config: ./model/config.json
  talker_weights: ./model/model.safetensors
  codec_config: ./codec/config.json
  codec_weights: ./codec/model.safetensors

generation_config:
  do_sample: true
  repetition_penalty: 1.05
  temperature: 0.9
  top_p: 1.0
  top_k: 50
  max_new_tokens: 8192
```

`generation_config.max_new_tokens` is the package default used when the CLI
does not receive an explicit `--max-new-tokens` override.

Run with:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --manifest ./path/to/qwen3_tts_package.yaml \
  --text "Hello from a custom layout." \
  --language English \
  --backend flex \
  --output ./out/base.wav
```

## CLI Help

```bash
cargo run --release -p tts_cli -- --help
cargo run --release -p tts_cli -- synthesize base --help
cargo run --release -p tts_cli -- synthesize custom-voice --help
```

## Troubleshooting

`invalid model directory`

- The directory is missing one of the required Hub files such as `config.json`
  or `generation_config.json`
- Point `--model-dir` at the actual model folder, not a higher-level parent

`unsupported language`

- The language name must exist in the model metadata
- Use a language label from the model card or `config.json`, such as `Chinese`
  or `English`

`unsupported speaker`

- The `--speaker` value must exist in the model metadata
- `--speaker` is only valid with `synthesize custom-voice`
- Base checkpoints typically do not contain any speaker list

No output file appears

- Check the final `--output` path
- The CLI creates parent directories automatically, but it still stops on model
  load or inference errors before writing audio
