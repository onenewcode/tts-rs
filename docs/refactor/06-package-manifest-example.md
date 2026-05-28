# Package Manifest Example

## Purpose

This document locks the v1 package manifest shape used by `tts_qwen3_tts`.
It exists to prevent the implementation phase from drifting back toward the old
workspace-level `models.yaml` registry design.

The file format is intentionally model-specific. That is a feature, not a
problem: the current refactor is about doing one model family correctly first.

## Example Manifest

Reference example: `docs/qwen3_tts_package.example.yaml`

```yaml
format: qwen3_tts_package/v1
name: qwen3-tts-12hz-0.6b-customvoice

artifacts:
  tokenizer: tokenizer.json
  talker_config: configs/talker.json
  talker_weights: weights/talker.safetensors
  codec_config: configs/codec.json
  codec_weights: weights/codec.safetensors

profiles:
  base:
    generation_config: profiles/base/generation_config.json
    control_config: profiles/base/control_config.json
  custom_voice:
    generation_config: profiles/custom_voice/generation_config.json
    control_config: profiles/custom_voice/control_config.json
```

## Serde Layer

The serde-facing structure should stay close to the file format:

```rust
pub struct Qwen3TtsPackageManifest {
    pub format: String,
    pub name: String,
    pub artifacts: Qwen3TtsArtifactsManifest,
    pub profiles: Qwen3TtsProfilesManifest,
}

pub struct Qwen3TtsArtifactsManifest {
    pub tokenizer: PathBuf,
    pub talker_config: PathBuf,
    pub talker_weights: PathBuf,
    pub codec_config: PathBuf,
    pub codec_weights: PathBuf,
}

pub struct Qwen3TtsProfilesManifest {
    pub base: Option<Qwen3TtsProfileManifest>,
    pub custom_voice: Option<Qwen3TtsProfileManifest>,
}

pub struct Qwen3TtsProfileManifest {
    pub generation_config: PathBuf,
    pub control_config: PathBuf,
}
```

The serde layer is allowed to be file-shape-oriented and stringly where needed.
It should not contain derived runtime objects.

## Normalized Package Layer

After parsing, the model crate converts manifest data into an internal package
fact object:

```rust
pub struct Qwen3TtsPackage {
    pub package_root: PathBuf,
    pub name: String,
    pub tokenizer_path: PathBuf,
    pub talker_config_path: PathBuf,
    pub talker_weights_path: PathBuf,
    pub codec_config_path: PathBuf,
    pub codec_weights_path: PathBuf,
    pub profiles: Qwen3TtsPackageProfiles,
}

pub struct Qwen3TtsPackageProfiles {
    pub base: Option<Qwen3TtsProfilePackage>,
    pub custom_voice: Option<Qwen3TtsProfilePackage>,
}

pub struct Qwen3TtsProfilePackage {
    pub generation_config_path: PathBuf,
    pub control_config_path: PathBuf,
}
```

Normalization responsibilities:

- resolve paths relative to the package root
- reject unsupported `format` values
- fail on missing required artifact entries
- preserve only package facts, not user defaults

## Why This Shape

Benefits:

- explicit and easy to validate
- no hidden coupling to workspace-level registries
- fixed profile fields keep compile paths simple
- future extraction to a more generic package layer remains possible if multiple
  model crates later converge on the same facts

Costs:

- intentionally model-specific today
- adding a new profile requires a schema change
- package manifests are not reusable as a cross-family registry format

These costs are acceptable because the current design target is correctness and
clarity for one model family, not premature generalization.

## Explicit Exclusions

The manifest must not contain:

- default speaker or language
- default backend
- default max token count
- profiling defaults
- output WAV path conventions
- model registry ids for other crates

Those are caller concerns or engine config concerns, not package facts.
