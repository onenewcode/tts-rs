# TTS Refactor Architecture

This repository is in the middle of a planned architecture reset. The current
source tree still contains the legacy `tts_core` + `tts_qwen` split, but that
layout is no longer the target design.

The implementation target is now:

- `tts_infer`: a thin inference-service crate that owns lifecycle orchestration,
  `PcmAudio`, and the internal session contract.
- `tts_qwen3_tts`: the concrete model crate for the current Qwen3-TTS model.
- `tts_cli`: a thin command-line entrypoint over `tts_qwen3_tts`.

The old design elements below are intended to be removed during implementation:

- the `tts_core` crate
- the public `variant/release` layer
- `tts_qwen/src/arch`
- fake-generic request types such as `text/language/speaker` in a shared core
- the old registry/executor facade path

Use the refactor documents under `docs/refactor/` as the source of truth for
implementation. They are intentionally written as target-state documents for
Goal-mode automation, not as descriptions of the current code.

## Refactor Document Set

- `docs/refactor/README.md`: overall objective, principles, and document index
- `docs/refactor/01-crate-boundaries.md`: workspace split and crate ownership
- `docs/refactor/02-tts-infer-contract.md`: thin service-layer contract and
  session state machine
- `docs/refactor/03-tts-qwen3-tts-design.md`: model-crate public API,
  package/compiler/model/session design
- `docs/refactor/04-cli-and-package-shape.md`: CLI shape and package-manifest
  responsibilities
- `docs/refactor/05-migration-and-acceptance.md`: migration map, acceptance
  criteria, and validation plan
- `docs/refactor/06-package-manifest-example.md`: locked target manifest shape
- `docs/refactor/07-source-tree-migration-map.md`: file-level migration map and
  target tree

## Example Inputs

- `docs/qwen3_tts_package.example.yaml`: target package manifest example
- `docs/models.example.yaml`: deprecated legacy registry example kept only as a
  marker that the old path is retired
