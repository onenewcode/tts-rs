# TTS Refactor Architecture

This repository now uses the refactored three-crate layout:

- `tts_infer`: a thin inference-service crate that owns lifecycle orchestration,
  `PcmAudio`, and the internal session contract.
- `tts_qwen3_tts`: the concrete model crate for the current Qwen3-TTS model.
- `tts_cli`: a thin command-line entrypoint over `tts_qwen3_tts`.

Key boundaries:

- `tts_infer` owns only service-level session orchestration and audio output.
- `tts_qwen3_tts` owns package parsing, request semantics, backend selection,
  compiler behavior, resident model state, and run-time session execution.
- `tts_cli` stays thin: parse package-first commands, map inputs into the model
  API, and write WAV output.

The refactor planning documents under `docs/refactor/` remain useful as the
design record for how this layout was reached.

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
