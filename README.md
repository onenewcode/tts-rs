# tts-rs

Rust workspace for running Qwen3-TTS locally and writing WAV output from the
CLI.

Workspace crates:

- `tts_infer`: thin inference/session layer
- `tts_qwen3_tts`: Qwen3-TTS loading, request compilation, and runtime
- `tts_cli`: command-line wrapper that writes `.wav` files

The CLI writes mono, 24 kHz, 16-bit PCM WAV output.

Important: run all CLI synthesis commands in release mode. Debug builds are
much slower and can look like they are hanging during model load or generation.

## Prerequisites

- Rust toolchain with Cargo
- A local Qwen3-TTS model directory downloaded from Hugging Face or copied from
  another machine

## Quickstart

The normal path is: pass the model directory directly.

Base model example:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from tts-rs." \
  --language English \
  --backend flex \
  --output ./out/base.wav
```

Custom voice example:

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --backend flex \
  --output ./out/custom-voice.wav
```

If you already built the binary:

```bash
target/release/tts_cli synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --backend flex \
  --output ./out/custom-voice.wav
```

Notes:

- `--model-dir` should point at the model folder itself, not its parent
- `./out/` is created automatically if it does not exist
- `--speaker` only applies to `synthesize custom-voice`
- language names are matched case-insensitively against the model metadata, so
  `Chinese` and `chinese` both work
- `--backend flex` is the recommended local default in this repository

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
- `Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice`

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

The custom-voice checkpoint in `Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice`
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
