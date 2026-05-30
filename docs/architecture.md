# TTS-RS Architecture

## Summary

`tts-rs` is being reshaped from a single-model local CLI project into a local,
on-device TTS framework that can host multiple heterogeneous model drivers.

The design center is not a unified cross-model request protocol. The design
center is a unified local model-service runtime:

- explicit driver registration
- explicit model loading
- loaded model instance lifecycle management
- single-instance serialized execution
- capability inspection after load
- shared diagnostics and framework-level management

The current codebase only ships a Qwen3-TTS driver, but the architecture must
leave room for additional model drivers with very different semantics.

## Current Workspace

Current workspace packages:

- `tts_core` (directory `tts_infer/`)
- `tts_error`
- `tts_qwen3_tts`
- `tts_app`
- `tts_cli`

Observed repo state:

- the framework core now lives in package `tts_core`, still rooted at `tts_infer/`
- `tts_qwen3_tts` still contains substantial model-private internals, but now exposes framework registration and loaded-instance capability projection
- `tts_cli` is thin and routes orchestration through `tts_app` instead of assembling rich driver requests itself

## Target Architecture

Target direction:

```text
frontend shell(s)
    |
    v
application services
    |
    v
framework core
    |
    +--> model driver: qwen3
    +--> model driver: future model A
    +--> model driver: future model B
```

Target workspace shape:

- `tts_core`
  local model-service framework core
- `tts_error`
  shared diagnostics foundation
- `tts_qwen3_tts`
  Qwen3-TTS driver crate
- `tts_app`
  application service orchestration for local frontends
- `tts_cli`
  CLI shell

Migration note:

- package `tts_core` currently lives in the existing `tts_infer/` directory to keep the repository move bounded while the API contract settles

## Core Principles

### 1. Framework before unified request protocol

The framework does not define a full cross-model synthesis request. Different
TTS models are expected to vary significantly in:

- prompting model
- speaker semantics
- reference audio rules
- supported outputs
- execution stages

The framework therefore standardizes lifecycle and management, not model input
semantics.

### 2. Loaded instances are the primary runtime object

The central managed entity is a loaded model instance.

Each instance:

- is explicitly created
- remains resident until explicitly closed
- is identified by a framework-assigned instance ID
- exposes capabilities after successful load
- executes one request at a time

### 3. Single-instance serialized execution

Each loaded instance is internally serialized.

Framework contract:

- instance states are `Ready`, `Busy`, `Closed`
- the handle layer guarantees serialization
- `close` rejects new work and waits for the in-flight request to finish
- an instance must be closed before the manager removes it

### 4. Capability inspection happens after load

Capabilities are not treated as fully reliable static metadata.

The authoritative capability view is derived from the loaded instance and its
resolved runtime configuration.

### 5. Workspace dependency versions are centralized

Shared dependency versions must remain declared in the workspace root
`Cargo.toml`.

Use `[workspace.dependencies]` for shared crates such as:

- `thiserror`
- `serde`
- `tracing`
- `clap`

Driver crates may still declare driver-specific dependencies locally, but
shared foundational versions must not drift per crate.

## Framework Core Contracts

The first framework revision should expose exactly these core public objects:

- `DriverDescriptor`
- `DriverRegistry`
- `LoadedModelHandle`
- `ModelManager`
- `ModelCapabilities`
- `SynthesisResult`

### DriverDescriptor

Purpose:

- describe a registered driver
- provide driver metadata
- declare the driver-specific load config type entrypoint

First revision fixed fields:

- `driver_id`
- `display_name`
- `summary`
- `config_type` or equivalent load-config type identity

Not included in v1:

- authoritative runtime capabilities
- per-model static discovery metadata

### DriverRegistry

Purpose:

- store registered drivers
- expose descriptors
- expose the creation entrypoint for manager-mediated loads

The registry manages drivers, not loaded instances.

### LoadedModelHandle

Purpose:

- represent one resident, loaded model instance
- expose minimal common management and inspection behavior

Common surface in v1:

- instance identity
- driver identity
- capabilities
- state summary
- close semantics

Important non-goal:

- no unified cross-model execute method

Actual synthesis is reached through model-specific handle downcast or model-
specific extension surface, not through a framework-wide request protocol.

### ModelManager

Purpose:

- load instances through registered drivers
- hold resident instances
- look them up by instance ID
- coordinate close/remove lifecycle

The manager owns instance indexing and lifetime control. It does not own model
discovery scanning.

### ModelCapabilities

Purpose:

- expose structured capability and routing information
- expose input constraints and output audio characteristics

Representation rule:

- fixed structure plus limited extension slot

Granularity rule:

- precise enough for framework routing, display, and validation boundaries
- not a full serialization of every model-private rule

### SynthesisResult

Purpose:

- return audio plus a minimal common execution summary

First revision contents:

- output audio
- `instance_id`
- `driver_id`
- elapsed execution time

Not included in v1:

- full execution statistics
- detailed backend telemetry

## Diagnostics

Error handling is split into:

- model-specific strong error types in each driver crate
- shared diagnostics infrastructure in `tts_error`

`tts_error` should contain:

- common error categories
- stable error codes
- diagnostic context container
- shared rendered diagnostic object

It should not become the home for every concrete model error enum.

## Qwen3 Driver Architecture

`tts_qwen3_tts` remains a single crate for now. It should not be split into
`tts_qwen3` and `tts_qwen3_model` by default.

Reason:

- the immediate reusable boundary is framework lifecycle management, not a
  generic "model semantics layer vs model execution layer" split
- future TTS models are expected to differ significantly
- forcing a second crate boundary too early risks encoding Qwen3-specific
  internals as platform structure

### Target Internal Subsystems

`tts_qwen3_tts` should be internally reorganized into:

- `surface/`
- `loading/`
- `capabilities/`
- `execution/`
- `backend/`

#### surface

Owns:

- public request types
- public run options
- public load options
- public engine or driver façade

Does not own:

- package normalization implementation
- request compilation internals
- capability aggregation internals
- runtime execution logic

#### loading

Owns:

- model-dir and manifest normalization
- package source handling
- config loading
- artifact validation
- loaded instance construction

#### capabilities

Owns:

- loaded-instance capability aggregation
- capability + constraint projection for framework inspection

#### execution

Owns:

- request pre-processing
- `reference-audio -> prompt` materialization
- request compilation
- runtime generation
- waveform production handoff

The current `compiler` layer is absorbed into `execution`.

#### backend

Owns:

- backend selection
- backend availability logic
- backend-specific boundary concerns

## Data Flow

Target Qwen3 request flow:

```text
surface request
-> execution pre-processing
-> execution compilation
-> execution runtime/generation
-> audio result
```

Target framework flow:

```text
register driver
-> manager loads instance through registry
-> loaded instance exposes capabilities
-> caller selects model-specific request path
-> model-specific synthesis runs under instance serialization
-> synthesis result returns audio + minimal metadata
```

## Non-Goals

Not part of the first refactor target:

- unified cross-model synthesis request
- plugin-style runtime discovery
- multi-request parallel execution within one loaded model instance
- automatic model-type detection from arbitrary paths
- centralized crate containing every concrete error enum

## Acceptance Criteria

Architecture documentation is acceptable only if it answers all of the
following unambiguously:

- what object the framework manages at runtime
- how loaded instances are identified
- how instance lifecycle transitions work
- whether capabilities are static or post-load
- whether the framework defines a unified execute request
- where Qwen3 request compilation belongs
- where `reference-audio -> prompt` materialization belongs
- how shared dependencies are versioned

