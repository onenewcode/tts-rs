# TTS-RS Architecture

## Summary

`tts-rs` is a Rust workspace for local, on-device text-to-speech inference.
The current implementation targets Qwen3-TTS and is built on Burn, but the
workspace is intentionally split so model loading, lifecycle control, request
preparation, and CLI concerns do not collapse into one crate.

Today the repository ships one concrete driver, `tts_qwen3_tts`, plus four
supporting crates around it:

- `tts_infer` (package name `tts_core`) for framework-level lifecycle and
  result primitives
- `tts_error` for shared diagnostics
- `tts_qwen3_tts` for the Qwen3-TTS driver and model-private runtime
- `tts_app` for application-service orchestration
- `tts_cli` for the thin command-line shell

At a high level, the architecture is:

```text
text -> request preparation -> talker/model -> codec tokens -> audio codec -> WAV
                           ^                                          |
                           +------ tts_core loaded-model lifecycle ---+
```

This split keeps the CLI small, keeps model-private details inside the driver
crate, and leaves room for more drivers without forcing every future model into
the exact same request type.

## Workspace Responsibilities

### `tts_infer` (`tts_core`)

`tts_infer` is the framework core. Its package name is `tts_core`, while the
directory remains `tts_infer/`.

It owns the reusable runtime contract:

- driver registration through `DriverRegistry`
- model loading through `ModelManager`
- loaded-instance handles through `LoadedModelHandle`
- lifecycle state tracking through `ModelState`
- shared capability projection through `ModelCapabilities`
- shared audio and synthesis output through `PcmAudio` and `SynthesisResult`

This crate does not define a universal cross-model synthesis request. That is
intentional: different TTS models can vary in prompt construction, speaker
selection, reference-audio requirements, and generation stages.

### `tts_error`

`tts_error` owns shared diagnostics and stable reporting structure. It is the
home for repository-wide error categories and user-facing diagnostic rendering,
without trying to absorb every model-specific enum into one crate.

### `tts_qwen3_tts`

`tts_qwen3_tts` is the current concrete driver. It owns:

- Qwen3-TTS public request and load surface
- package normalization and artifact loading
- capability aggregation after load
- request compilation and execution
- model-private talker, speaker, codec, and neural-network internals

This crate is the Burn-backed runtime center of the repository today.

### `tts_app`

`tts_app` is the application-service layer between shells and drivers. It owns:

- shell-facing synthesis input structs
- translation from CLI semantics into driver-facing requests
- package source selection (`--model-dir` vs `--manifest`)
- profiling and runtime option assembly
- load / synthesize / save orchestration through `ModelManager`

This keeps request assembly and validation out of `tts_cli`.

### `tts_cli`

`tts_cli` is intentionally thin. It should:

- parse command-line arguments
- map flags into `tts_app` input structs
- call `tts_app`
- report output paths and diagnostics

It should not grow model-specific runtime logic or duplicate request-building
rules that belong in `tts_app`.

## Runtime Model

The central runtime object is a loaded model instance managed by `tts_core`.

Current framework flow:

```text
register driver
-> manager loads instance through registry
-> loaded instance exposes capabilities
-> caller uses model-specific execution surface
-> synthesis runs under per-instance serialization
-> synthesis result returns audio plus runtime metadata
```

### Driver Registration

Drivers are registered explicitly in a `DriverRegistry`. The current app path
registers the Qwen3 driver during `QwenAppService::new()`.

This is a deliberate choice:

- driver availability is explicit
- startup behavior is deterministic
- framework code does not depend on plugin-style discovery

### Loaded Instances And Lifecycle

`ModelManager` loads resident model instances and returns a
`LoadedModelHandle`.

Each loaded instance:

- has a framework-assigned instance ID
- exposes capabilities after successful load
- executes one request at a time
- must be closed before removal

The current per-instance state model is:

- `Ready`
- `Busy`
- `Closed`

Serialized execution is enforced at the handle layer so one loaded model
instance cannot process overlapping requests accidentally.

### Capability Inspection

Capabilities are resolved after load instead of being treated as purely static
metadata. The loaded artifacts and resolved runtime configuration determine the
authoritative `ModelCapabilities` view exposed through the framework.

## Current Qwen3 Driver Layout

`tts_qwen3_tts/src/` is currently organized into these primary layers:

- `surface/`
- `loading/`
- `capabilities/`
- `execution/`
- `model/`

There is also a small top-level `sampling.rs` module for reusable sampling
configuration types.

### `surface/`

Owns the public driver-facing API:

- request types such as `BaseRequest`, `CustomVoiceRequest`, and `QwenRequest`
- package and manifest surface types
- load and run options
- language and prompt surface enums
- driver registration entrypoints

If another crate needs to talk to the Qwen3 driver directly, it should prefer
this layer rather than reaching into model-private modules.

### `loading/`

Owns model package normalization and load-time preparation:

- model-dir vs manifest handling
- package manifest parsing
- relative-path normalization
- artifact validation
- generation-config loading
- construction of the loaded Qwen3 engine

This layer is where repository-local file layout assumptions are turned into a
driver runtime.

### `capabilities/`

Owns loaded-instance capability projection for framework consumers. This keeps
the framework-facing capability shape separate from the lower-level loading and
execution details that produce it.

### `execution/`

Owns runtime request handling after load:

- request compilation
- prompt construction
- session seed and sampling resolution
- profiling configuration
- reference-audio preparation
- generation loop execution
- audio finalization

This is the operational pipeline from driver-facing request to synthesized
audio.

### `model/`

Owns model-private neural and signal-processing internals:

- `talker/`
- `speaker/`
- `codec/`
- `nn/`

This tree is intentionally not the public framework surface. It is where the
Qwen3 implementation details live, including Burn module wiring and tensor
operations.

## Burn And Backend Shape

The current runtime is built on Burn. Backend selection is surfaced through the
Qwen3 driver and propagated through `tts_app` and `tts_cli`.

Current feature flags include:

- `flex`
- `fusion`
- `wgpu`
- `cuda`
- `rocm`
- `metal`
- `vulkan`
- `webgpu`

In practice, this means:

- backend choice is part of the driver/application boundary
- CLI commands stay backend-aware without embedding backend internals
- CI and linting should be careful with `--all-features`, because some optional
  backend stacks require extra platform SDKs

The repository's recommended local default remains `flex`.

## End-To-End Request Flow

The current CLI-to-audio path is:

```text
tts_cli args
-> tts_app input structs
-> package source resolution
-> ModelManager loads Qwen3 instance
-> qwen3 request compilation
-> talker generation and codec/audio finalization
-> WAV output
```

More concretely:

1. `tts_cli` parses shell flags.
2. `tts_app` converts those flags into `BaseSynthesisInput` or
   `CustomVoiceSynthesisInput`.
3. `tts_app` builds a `QwenRequest`, `Qwen3TtsRunOptions`, and
   `Qwen3TtsProfilingConfig`.
4. `ModelManager` loads a Qwen3 instance through the registered driver.
5. `Qwen3TtsHandleExt` runs synthesis on the loaded handle.
6. The resulting `PcmAudio` is saved as WAV.
7. The handle is closed and removed from the manager.

The current app service is intentionally request-scoped: it loads, uses, saves,
closes, and removes the model instance within the synthesis path.

## Current Architectural Boundaries

The codebase currently aims to keep these boundaries clear:

- `tts_cli` stays a shell, not an application-service crate
- `tts_app` owns request assembly and shell-to-driver translation
- `tts_infer` owns reusable lifecycle primitives, not model semantics
- `tts_qwen3_tts` owns Qwen3-specific execution and Burn integration
- `tts_error` owns shared diagnostics, not every possible runtime detail

## Current Non-Goals

The current architecture does not aim to provide:

- a unified cross-model synthesis request type
- plugin-style runtime discovery
- parallel request execution inside one loaded model instance
- automatic support for arbitrary model layouts without explicit packaging rules
- a fully generic backend abstraction outside the current driver surface

## Target Direction

The near-term direction remains a layered local runtime:

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

Near-term priorities:

- keep `tts_cli` thin
- continue moving shell semantics into `tts_app`
- preserve the `surface` / `loading` / `capabilities` / `execution` split in
  `tts_qwen3_tts`
- keep model-private Burn internals inside `model/`
- only extract a stronger backend boundary when the implementation shape makes
  that split concretely useful
