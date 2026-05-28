# CLI And Package Shape

## CLI Direction

`tts_cli` becomes package-first. It does not keep the old global
`models.yaml + model_id` registry shape in v1.

The CLI must:

- accept a package path (manifest path or package directory)
- select profile via subcommands
- build `QwenRequest`
- build `Qwen3TtsRunOptions`
- call `Qwen3TtsEngine::load(...).synthesize(...)`
- write `PcmAudio` to WAV

## Profile Selection

CLI profile selection must use subcommands rather than a `--profile` flag.

Target shape:

```text
tts_cli synthesize base ...
tts_cli synthesize custom-voice ...
```

Reasoning:

- matches `QwenRequest` enum shape
- keeps profile-specific parameters from collapsing back into one large set of
  optional flags
- makes future profile additions explicit

## Exact Command Shape

The CLI contract should be concrete enough that later implementation does not
need to invent flag semantics.

```text
tts_cli synthesize base \
  --package ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello" \
  --language auto \
  --output out.wav
```

```text
tts_cli synthesize custom-voice \
  --package ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "Hello" \
  --language zh \
  --speaker Chelsie \
  --output out.wav
```

Shared flags for both profile subcommands:

- `--package <PATH>`: required; accepts manifest path or package directory
- `--text <TEXT>`: required in v1; stdin streaming is not part of this refactor
- `--language <auto|NAME>`: optional; defaults to `auto`
- `--output <PATH>`: required; target WAV path
- `--backend <BACKEND>`: optional; parsed as `Qwen3TtsBackend`, default is
  runtime resolution from `Qwen3TtsBackend::Flex`
- `--max-new-tokens <N>`: optional; default `256`
- `--sampling <MODE>`: optional; v1 only needs `greedy`, but the parser should
  still map into `SamplingConfig`
- `--profiling`: optional bool; enables profile logging
- `--profiling-per-step`: optional bool; requests per-step logs
- `--profiling-stage-summary/--no-profiling-stage-summary`: optional explicit
  stage-summary toggle
- `--profiling-log-topk <N>`: optional; default `8`

Profile-specific flags:

- `base`: no extra flags
- `custom-voice`: `--speaker <NAME>` optional

Non-goals for the v1 CLI:

- no `model_id`
- no `variant`
- no `release`
- no global `models.yaml`
- no `--profile` string selector
- no streaming chunk output flags

## CLI To Model Mapping

`tts_cli` is only an adapter. Mapping rules are fixed:

- `synthesize base` -> `QwenRequest::Base(BaseRequest { .. })`
- `synthesize custom-voice` ->
  `QwenRequest::CustomVoice(CustomVoiceRequest { .. })`
- `--language auto` -> `LanguageSelection::Auto`
- `--language <NAME>` -> `LanguageSelection::Named(NAME.to_owned())`
- `--max-new-tokens` + `--sampling` -> `Qwen3TtsRunOptions`
- profiling flags -> `Qwen3TtsProfilingConfig`
- `--package` -> `Qwen3TtsPackageSource::{ManifestPath, PackageDir}` after path
  normalization

This mapping is intentionally boring. `tts_cli` must not contain package
inspection logic, prompt decisions, or backend feature-policy logic.

## Package Source

```rust
pub enum Qwen3TtsPackageSource {
    ManifestPath(PathBuf),
    PackageDir(PathBuf),
}
```

This keeps external package selection flexible while letting the model crate
normalize both forms into one internal `Qwen3TtsPackage`.

## Manifest Split

Two layers are required:

- `Qwen3TtsPackageManifest`: serde-facing file format
- `Qwen3TtsPackage`: internal normalized fact object

The manifest file format must not be the same thing as the engine's internal
package object.

## Package Facts

`Qwen3TtsPackage` stores normalized paths and supported profile facts only.
It must not store:

- default speaker
- default language
- default run options
- CLI output preferences
- user-facing presets

## Profile Facts In Package

`Qwen3TtsPackage.profiles` records:

- whether `base` is supported
- whether `custom_voice` is supported
- for each supported profile, the package-level paths needed by the compiler

That means profile package entries contain fact paths such as:

- `generation_config_path`
- `control_config_path`

The package layer does not own parsed config objects; those are loaded into the
compiler during engine initialization.

See `docs/refactor/06-package-manifest-example.md` and
`docs/qwen3_tts_package.example.yaml` for the locked target manifest shape.
