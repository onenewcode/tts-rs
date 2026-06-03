# tts_cli

`tts_cli` is the command-line entrypoint for this workspace. It keeps argument
parsing and output handling in the CLI layer while routing request preparation
and model execution through `tts_app`.

This document covers the current CLI behavior in the `tts_cli` crate itself.
The examples here match the live `--help` output and avoid stale flags that are
no longer accepted.

## What It Does

`tts_cli` exposes one top-level workflow:

```bash
tts_cli synthesize <profile> [options]
```

Current synthesis profiles:

- `base`: standard Qwen3-TTS base generation and base-model voice cloning
- `custom-voice`: synthesis against a custom-voice checkpoint with a named
  speaker and optional speaking-style instruction

The CLI writes PCM WAV output and creates the parent output directory if it
does not already exist.

## Prerequisites

- Rust toolchain with Cargo
- A local Qwen3-TTS model package
- Enough memory and time to run release builds; debug builds are much slower

Common repo-local model directories:

- `./Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

## Build And Run

Run the CLI in release mode:

```bash
cargo run --release -p tts_cli -- synthesize --help
```

If you need a non-default runtime backend, select it through Cargo features at
build time, not through a CLI flag. For example:

```bash
cargo run --release -p tts_cli --no-default-features --features cuda -- synthesize --help
```

Important: the current CLI does not accept `--backend`. Older examples in other
documents may still show it, but `tts_cli` now chooses runtime support through
crate features such as `flex`, `fusion`, `cuda`, `wgpu`, `metal`, or `vulkan`.

## Command Shape

```bash
tts_cli synthesize base [OPTIONS] --text <TEXT> --output <OUTPUT>
tts_cli synthesize custom-voice [OPTIONS] --text <TEXT> --output <OUTPUT> --speaker <SPEAKER>
```

Shared options for both profiles:

- `--model-dir <MODEL_DIR>`: path to the model directory
- `--manifest <MANIFEST>`: path to a `qwen3_tts_package.yaml` file for
  non-standard layouts
- `--text <TEXT>`: target text to synthesize
- `--language <LANGUAGE>`: language name; defaults to `auto`
- `--output <OUTPUT>`: WAV output path
- `--sampling <SAMPLING>`: currently only `greedy`
- `--log-level <LOG_LEVEL>`: one of `error`, `warn`, `info`, `debug`, `trace`
- profiling flags: `--profiling`, `--profiling-per-step`,
  `--profiling-stage-summary`, `--no-profiling-stage-summary`,
  `--profiling-log-topk`

Rules:

- pass exactly one of `--model-dir` or `--manifest`
- `--model-dir` should point at the model folder itself, not its parent
- language matching is case-insensitive against model metadata

## Base Profile

Use `base` for normal synthesis with a base checkpoint, or for voice-clone
flows driven by reference audio.

### Plain Base Synthesis

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the base profile." \
  --language English \
  --output ./out/base_plain.wav
```

### Voice Clone With `ref_audio + ref_text`

This is the in-context learning clone path. `--ref-text` must be the transcript
of `--ref-audio`, not the target text.

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone ICL smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --ref-text "Hello from the generated reference clip." \
  --output ./out/base_clone_icl_release.wav
```

### Voice Clone With `--x-vector-only`

This mode uses only the speaker embedding from the reference audio.

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone x-vector-only smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --x-vector-only \
  --output ./out/base_clone_xvector_release.wav
```

Reference-audio rules:

- `--ref-text` requires `--ref-audio`
- `--x-vector-only` requires `--ref-audio`
- `--x-vector-only` conflicts with `--ref-text`
- if you want better ICL clone quality, use the real transcript of the
  reference clip

## Custom-Voice Profile

Use `custom-voice` with a custom-voice checkpoint that exposes named speakers.

### Basic Custom-Voice Synthesis

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --output ./out/custom-voice.wav
```

### Custom-Voice Synthesis With `--instruct`

`--instruct` describes the intended speaking style for the target text.

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --output ./out/custom-voice-instruct.wav
```

For the checked-in custom-voice model in this repository, common speakers
include:

- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ryan`
- `Aiden`
- `Ono_Anna`
- `Sohee`

The actual supported speaker list comes from model metadata. If you pass an
unsupported speaker, the CLI reports what the loaded model supports.

## Manifest Mode

Most users should prefer `--model-dir`. Use `--manifest` only when your files
do not follow the default model directory layout.

Example:

```bash
cargo run --release -p tts_cli -- synthesize base \
  --manifest ./path/to/qwen3_tts_package.yaml \
  --text "Hello from a custom manifest layout." \
  --language English \
  --output ./out/base_manifest.wav
```

## Troubleshooting

`unexpected argument '--backend' found`

- The current CLI no longer accepts `--backend`
- Choose backend support through Cargo features instead

`the following required arguments were not provided`

- Check whether you omitted `--text`, `--output`, or `--speaker`
- For `custom-voice`, `--speaker` is required

`--ref-text is required when --ref-audio is used without --x-vector-only`

- Base ICL clone requires both the reference audio and its transcript

`unsupported speaker`

- The `--speaker` value must exist in the loaded model metadata
- Base checkpoints usually do not expose named speaker lists

Slow startup or generation

- Use `--release`
- Large model loads can take noticeable time before generation starts

## Useful Help Commands

```bash
cargo run --release -p tts_cli -- synthesize --help
cargo run --release -p tts_cli -- synthesize base --help
cargo run --release -p tts_cli -- synthesize custom-voice --help
```
