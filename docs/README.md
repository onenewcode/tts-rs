# Documentation Index

## Summary

This directory contains the target-state architecture and refactor documents for
the local on-device TTS framework migration.

Read these documents in order when planning or reviewing architectural work.

## Primary Documents

- [architecture.md](/Volumes/mian/code/rs/tts-rs/docs/architecture.md)
  source-of-truth architecture overview and core design principles
- [testing_tts_qwen.md](/Volumes/mian/code/rs/tts-rs/docs/testing_tts_qwen.md)
  test layering, verification commands, and smoke-test policy

## Refactor Documents

- [01-current-state-audit.md](/Volumes/mian/code/rs/tts-rs/docs/refactor/01-current-state-audit.md)
  current-repo audit and architectural pressure points
- [02-target-architecture.md](/Volumes/mian/code/rs/tts-rs/docs/refactor/02-target-architecture.md)
  target crate map, dependency direction, and Qwen3 driver layering
- [03-migration-plan.md](/Volumes/mian/code/rs/tts-rs/docs/refactor/03-migration-plan.md)
  staged migration sequence, exit criteria, and verification gates
- [04-api-spec.md](/Volumes/mian/code/rs/tts-rs/docs/refactor/04-api-spec.md)
  first-revision API contract baseline for core and Qwen3 driver boundaries

## Intended Usage

Use these documents as follows:

- use `architecture.md` to answer "what system are we building"
- use `02-target-architecture.md` to answer "where does this responsibility go"
- use `04-api-spec.md` to answer "what is the intended public contract"
- use `03-migration-plan.md` to answer "what order should we change things in"
- use `testing_tts_qwen.md` to answer "how do we verify each stage"

## Acceptance Checklist

This documentation set is complete only when:

- the architecture overview exists
- the testing strategy exists
- the current-state audit exists
- the target architecture exists
- the migration plan exists
- the API baseline exists

