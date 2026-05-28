# `tts_infer` Contract

## Role

`tts_infer` is a thin service layer. It owns lifecycle orchestration, not model
semantics.

## Shared Result Type

```rust
pub struct PcmAudio {
    pub pcm_i16: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
}
```

`PcmAudio` is intentionally shared because it is stable enough across model
implementations and directly serves CLI and future service outputs.

## Internal Session Contract

```rust
pub enum SessionStep {
    Advanced,
    Finished,
}

pub enum ServiceError {
    StepAfterTerminal,
    FinishBeforeTerminal,
}

pub enum InferError<E> {
    Model(E),
    Service(ServiceError),
}

pub trait LoadedModel {
    type Request;
    type RunOptions;
    type Session: ModelSession<Error = Self::Error>;
    type Error;

    fn start_session(
        &self,
        request: Self::Request,
        options: Self::RunOptions,
    ) -> Result<Self::Session, Self::Error>;
}

pub trait ModelSession {
    type Error;

    fn step(&mut self) -> Result<SessionStep, Self::Error>;
    fn finish(self) -> Result<PcmAudio, Self::Error>;
}
```

## Engine Skeleton

```rust
pub struct Engine<M> {
    model: M,
}

pub struct EngineSession<S> {
    inner: S,
    state: EngineSessionState,
}

enum EngineSessionState {
    Running,
    TerminalReached,
}
```

```rust
impl<M: LoadedModel> Engine<M> {
    pub fn new(model: M) -> Self;

    pub fn synthesize(
        &self,
        request: M::Request,
        options: M::RunOptions,
    ) -> Result<PcmAudio, InferError<M::Error>>;

    pub fn start_session(
        &self,
        request: M::Request,
        options: M::RunOptions,
    ) -> Result<EngineSession<M::Session>, InferError<M::Error>>;
}
```

## Session State Guard Rules

`EngineSession<S>` must guard the service-level session state machine:

- `step()` is valid only while the wrapper state is `Running`
- when the model session returns `SessionStep::Finished`, wrapper state flips to
  `TerminalReached`
- calling `step()` after terminal state returns
  `InferError::Service(ServiceError::StepAfterTerminal)`
- `finish(self)` is valid only when wrapper state is `TerminalReached`
- calling `finish(self)` before terminal state returns
  `InferError::Service(ServiceError::FinishBeforeTerminal)`

`finish(self)` is not a partial-result API. Future early-stop semantics must use
new methods such as `cancel(self)` or `finish_partial(self)` rather than
changing `finish()` semantics.

## Non-Goals For `tts_infer`

- no generic package format
- no generic backend capability system
- no generic TTS request schema
- no platform-wide profiling registry
- no dynamic trait-object plugin system
- no thread-safety bounds baked into v1 traits
