# TTS-RS Architecture

## Summary

`tts-rs` is a local, on-device TTS workspace built around explicit model-driver
registration and loaded-instance lifecycle management. The repository currently
ships one concrete driver, `tts_qwen3_tts`, but the surrounding runtime is
structured so additional drivers can be added without forcing all models into a
single synthesis request shape.

The current design center is a shared local model-service runtime:

- explicit driver registration
- explicit model loading
- loaded model instance lifecycle management
- single-instance serialized execution
- capability inspection after load
- shared diagnostics and framework-level orchestration

The framework does not define a fully unified cross-model synthesis request.
Different TTS models are expected to vary in prompting, speaker semantics,
reference-audio rules, and execution stages.

## Current Workspace Shape

The current workspace contains five crates:

- `tts_core` (directory `tts_infer/`)
- `tts_error`
- `tts_qwen3_tts`
- `tts_app`
- `tts_cli`

Current responsibilities:

- `tts_core` provides the framework-level manager/handle layer and common audio,
  capability, and result primitives
- `tts_error` provides shared diagnostics and stable error-reporting structure
- `tts_qwen3_tts` implements the Qwen3-TTS driver and its model-private runtime
- `tts_app` prepares CLI-facing requests into driver-facing requests and owns
  application-service orchestration
- `tts_cli` is a thin shell over `tts_app`

The package name for the framework crate is already `tts_core`, even though the
source directory remains `tts_infer/`.

## Runtime Model

The central runtime object is a loaded model instance managed through
`tts_core`.

Current public framework objects include:

- `DriverDescriptor`
- `DriverRegistry`
- `LoadedModelHandle`
- `ModelManager`
- `ModelCapabilities`
- `SynthesisResult`

These define the framework contract that the rest of the workspace builds on.

### Driver registration and loading

Drivers are registered explicitly in a `DriverRegistry`. A `ModelManager` owns
the registry-backed loading path and creates resident model instances through
that registry.

In the current application path, `tts_app::QwenAppService`:

- constructs a registry
- registers the Qwen3 driver
- loads a model instance through `ModelManager`
- calls the Qwen3-specific synthesis extension on the loaded handle
- closes the handle and removes the instance after synthesis completes

### Loaded instances and lifecycle

Loaded instances are the primary managed object.

Each instance:

- is explicitly created through the manager
- receives a framework-assigned instance ID
- exposes capabilities after successful load
- executes one request at a time
- must be closed before the manager removes it

The handle layer enforces serialized per-instance work. The current state model
is `Ready`, `Busy`, and `Closed`.

### Capability inspection

Capabilities are authoritative after load, not only from static package
metadata. The loaded instance and resolved runtime configuration determine the
capability view surfaced through `ModelCapabilities`.

### Diagnostics

Error handling is split between:

- model-specific error types in the driver crate
- shared diagnostics infrastructure in `tts_error`

`tts_error` is the home for shared categories, codes, and rendered diagnostics;
it is not intended to absorb every model-specific error enum.

## Current Qwen3 Driver Structure

`tts_qwen3_tts` remains a single crate. The immediate reusable boundary in this
repository is framework lifecycle management rather than a second crate split
between generic model semantics and model execution.

The current top-level internal areas are:

- `surface/`
- `loading/`
- `capabilities/`
- `execution/`
- `model/`

### `surface/`

Owns the public driver-facing request types, load options, run options, backend
selection surface, and registration entrypoints re-exported from the crate.

### `loading/`

Owns package normalization, config loading, artifact validation, and loaded
instance construction.

### `capabilities/`

Owns loaded-instance capability aggregation and projection for framework-facing
inspection.

### `execution/`

Owns request pre-processing, reference-audio prompt materialization, request
compilation, runtime generation, profiling hooks, and waveform production
handoff.

### `model/`

Holds model-private internals such as talker, speaker, codec, and neural-network
subsystems. This tree still carries substantial implementation detail that is
not part of the framework surface.

Backend-specific concerns exist today, but they are not yet factored into a
standalone top-level `backend/` directory in the current source tree.

## Data Flow

Current CLI-to-audio flow:

```text
tts_cli args
-> tts_app request preparation
-> ModelManager loads Qwen3 instance
-> qwen3 execution pipeline compiles and generates
-> audio is written to WAV
```

Current framework flow:

```text
register driver
-> manager loads instance through registry
-> loaded instance exposes capabilities
-> caller uses model-specific execution surface
-> synthesis runs under per-instance serialization
-> synthesis result returns audio plus runtime metadata
```

## Current Non-Goals

The current architecture does not aim to provide:

- a unified cross-model synthesis request
- plugin-style runtime discovery
- multi-request parallel execution inside one loaded model instance
- automatic model-type detection from arbitrary paths
- a centralized crate containing every concrete error enum

## Target Direction

The next-step direction remains a layered local runtime:

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

Target-direction notes:

- keep `tts_cli` thin and move request-assembly semantics into `tts_app`
- continue tightening the split between Qwen3 public surface, loading,
  capability projection, and execution internals
- extract backend boundary concerns more explicitly only when the current module
  shape makes that split useful rather than aspirational
