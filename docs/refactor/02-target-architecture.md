# Target Architecture

## Summary

The target architecture is a local on-device TTS framework with explicit model
drivers, not a single-model application and not a unified cross-model request
platform.

The stable shared boundary is model-service management. Model semantics remain
driver-specific.

## Target Workspace

Target crates:

- `tts_core`
- `tts_error`
- `tts_qwen3_tts`
- `tts_app`
- `tts_cli`

## Dependency Direction

Target dependency direction:

```text
tts_cli -> tts_app -> tts_core
                   -> tts_error

tts_app -> tts_qwen3_tts
tts_qwen3_tts -> tts_core
tts_qwen3_tts -> tts_error
```

Rules:

- frontend shells depend downward only
- drivers do not depend on frontend shells
- framework core does not depend on model-specific request types
- shared dependency versions are declared in the workspace root

## Framework Core Public Objects

First revision public set:

- `DriverDescriptor`
- `DriverRegistry`
- `LoadedModelHandle`
- `ModelManager`
- `ModelCapabilities`
- `SynthesisResult`

### DriverDescriptor

Fixed fields:

- `driver_id`
- `display_name`
- `summary`
- config type identity for driver-specific load config

### DriverRegistry

Responsibilities:

- hold registered drivers
- expose descriptors
- expose creation entrypoints used by the manager

### LoadedModelHandle

Responsibilities:

- expose instance identity
- expose driver identity
- expose capabilities
- expose lifecycle state
- expose close semantics

State model:

- `Ready`
- `Busy`
- `Closed`

Execution policy:

- handle layer serializes access
- no unified execute request in framework core
- synthesis occurs through driver-specific handle downcast or driver extension

### ModelManager

Responsibilities:

- load instances through registered drivers
- hold resident instances
- look up resident instances
- close resident instances
- remove only already closed instances

Not responsible for:

- filesystem discovery
- auto-detecting model type from arbitrary path

### ModelCapabilities

Representation:

- fixed structured fields
- limited extension slot

Contents:

- supported high-level abilities
- routing-relevant input constraints
- output audio characteristics

Source of truth:

- loaded instance

### SynthesisResult

Minimum fields:

- audio output
- instance ID
- driver ID
- elapsed time

## Shared Diagnostics

`tts_error` responsibilities:

- error categories
- stable error codes
- shared diagnostic context
- rendered diagnostic envelope

Non-goal:

- centralizing all concrete model errors in one crate

## Qwen3 Driver Shape

`tts_qwen3_tts` remains single-crate for the first major refactor.

Internal layers:

- `surface`
- `loading`
- `capabilities`
- `execution`
- `backend`

### surface

Public content:

- requests
- run options
- load options
- façade

No internal execution protocols exposed here.

### loading

Owns:

- package source normalization
- manifest/model-dir parsing
- config loading
- artifact validation
- loaded instance construction

### capabilities

Owns:

- loaded-instance capability aggregation
- capability projection into framework-friendly structured form

### execution

Owns:

- request pre-processing
- reference-audio prompt materialization
- request compilation
- generation
- waveform/PCM handoff

The current `compiler` layer moves here.

### backend

Owns:

- backend selection
- backend availability checks
- backend-facing integration boundaries

## Frontend Layer Goals

### tts_app

Purpose:

- absorb orchestration now living in CLI
- provide reusable application services for local frontends

### tts_cli

Purpose:

- parse arguments
- call application services
- report output paths and diagnostics

Must not remain responsible for:

- core synthesis semantics
- cross-field request assembly rules
- package resolution policy

## Acceptance Criteria

This target architecture doc is acceptable only if it unambiguously states:

- the target crate list
- dependency direction
- the six core public framework objects
- the lifecycle state model
- that there is no unified execute request in `tts_core`
- the five internal Qwen3 sublayers

