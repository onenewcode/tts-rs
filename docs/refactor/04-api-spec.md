# API Specification Baseline

## Summary

This document defines the intended first revision public API boundary for the
future framework core and the reshaped Qwen3 driver.

It is not an implementation listing. It is a contract guide for future code
movement and API design.

## Framework Core API

### DriverDescriptor

Purpose:

- identify one registered driver
- present minimal operator-facing driver metadata
- describe the driver-specific load-config entrypoint

Required fields:

- `driver_id`
- `display_name`
- `summary`
- `config_type`

Required guarantees:

- `driver_id` is stable within the workspace
- descriptor access does not require model loading

### DriverRegistry

Purpose:

- register drivers explicitly
- list descriptors
- provide creation entrypoints to the manager

Required behavior:

- duplicate `driver_id` registration must be rejected
- descriptor listing must be deterministic

### LoadedModelHandle

Purpose:

- represent one resident loaded instance

Required common interface:

- get instance ID
- get driver ID
- inspect capabilities
- inspect lifecycle state
- begin close

Lifecycle contract:

- `Ready` accepts synthesis work
- `Busy` represents one in-flight request
- `Closed` rejects new work

Close contract:

- closing rejects new work
- closing waits for the current request to finish
- a closed instance may be removed by the manager

Execution contract:

- no framework-wide typed synthesis request is defined
- actual synthesis is reached through model-specific handle downcast or
  equivalent driver-specific extension

### ModelManager

Purpose:

- create loaded instances through registered drivers
- retain them by framework-assigned instance ID
- expose lookup and removal APIs

Required behavior:

- loading requires explicit driver selection
- removal of non-closed instances must be rejected
- instance IDs are framework-owned and unique

### ModelCapabilities

Purpose:

- provide framework-visible ability and constraint summary

Required shape:

- structured core fields
- limited extension slot

Required information classes:

- supported top-level abilities
- routing-relevant input constraints
- output audio properties

Source of truth:

- post-load instance aggregation

### SynthesisResult

Purpose:

- return generated audio and minimal shared execution metadata

Required fields:

- audio
- `instance_id`
- `driver_id`
- elapsed duration

## Qwen3 Driver Public API

### Surface exports

The public Qwen3 surface should expose:

- request types
- run options
- load options
- public façade or engine entrypoint

It should not expose:

- execution intermediate protocols
- compilation intermediate types
- backend-private runtime structs

### Request family

The Qwen3 driver keeps model-specific request semantics.

Current request families:

- base request
- custom voice request
- voice-clone conditioning

Rule:

- these remain driver-specific, not promoted into framework core

### Loading API

Qwen3 keeps driver-specific typed load configuration.

Rule:

- no conversion to generic JSON/YAML value config in the framework core

### Capabilities API

The Qwen3 driver must project a loaded-instance capability view that can be
consumed by `tts_core` while still keeping model-private detail internal.

## Validation Rules

### Framework-level validation

Framework-level validation should cover:

- driver registration validity
- instance lifecycle legality
- closed-versus-removable semantics

### Driver-level validation

Driver-level validation should cover:

- request semantic legality
- package/model-dir validity
- profile support
- voice-clone input requirements

### CLI-level validation

CLI-level validation should cover:

- parse-level shell constraints only

## Acceptance Criteria

This API spec is acceptable only if it clearly states:

- the required fields of the six framework objects
- that execution is not unified in `tts_core`
- that Qwen3 requests remain driver-specific
- that load configs stay strongly typed per driver

