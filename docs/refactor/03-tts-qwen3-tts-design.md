# `tts_qwen3_tts` Design

## Public API

```rust
pub struct Qwen3TtsEngine {
    inner: tts_infer::Engine<Qwen3TtsLoadedModel>,
}

impl Qwen3TtsEngine {
    pub fn load(config: Qwen3TtsEngineConfig) -> Result<Self, Qwen3TtsError>;

    pub fn synthesize(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<tts_infer::PcmAudio, Qwen3TtsError>;
}
```

V1 does not expose a public session API, but internally the crate must already
use the session-based seam from `tts_infer`.

## Request Types

```rust
pub enum QwenRequest {
    Base(BaseRequest),
    CustomVoice(CustomVoiceRequest),
}

pub enum LanguageSelection {
    Auto,
    Named(String),
}

pub struct BaseRequest {
    pub text: String,
    pub language: LanguageSelection,
}

impl BaseRequest {
    pub fn new(text: impl Into<String>) -> Self;
}

pub struct CustomVoiceRequest {
    pub text: String,
    pub language: LanguageSelection,
    pub speaker: Option<String>,
}

impl CustomVoiceRequest {
    pub fn new(text: impl Into<String>) -> Self;
}
```

Semantic rules:

- `BaseRequest` does not support `speaker`
- `CustomVoiceRequest` supports optional `speaker`
- `LanguageSelection::Auto` replaces the old implicit `None => "auto"` logic

## Engine Config

```rust
pub struct Qwen3TtsEngineConfig {
    pub package: Qwen3TtsPackageSource,
    pub backend: Qwen3TtsBackend,
    pub profiling: Qwen3TtsProfilingConfig,
}
```

```rust
pub struct Qwen3TtsRunOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}
```

`Qwen3TtsRunOptions::default()` is locked to:

- `max_new_tokens = 256`
- `sampling = SamplingConfig::greedy()`

## Backend API

```rust
pub enum Qwen3TtsBackend {
    Flex,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    WebGpu,
}
```

The backend module keeps one true backend enum and owns:

- `label()`
- `available()`
- `resolve(selected)`
- `FromStr`

No second backend enum should survive in another crate.

## Profiling Config

```rust
pub struct Qwen3TtsProfilingConfig {
    pub enabled: bool,
    pub per_step: bool,
    pub stage_summary: bool,
    pub log_topk: usize,
}
```

Defaults are locked to:

- `enabled = false`
- `per_step = false`
- `stage_summary = true`
- `log_topk = 8`

## Resident Model Objects

```rust
struct Qwen3TtsModelInner<B: Backend> {
    package: Qwen3TtsPackage,
    device: B::Device,
    compiler: Qwen3TtsRequestCompiler,
    talker: LoadedQwen3TtsTalker<B>,
    decoder: LoadedQwen3TtsAudioCodec<B>,
}
```

`device/backend` belongs to `ModelInner` because it is part of the resident
execution environment.

### Loaded Model Erasure

```rust
enum Qwen3TtsLoadedModel {
    #[cfg(feature = "flex")]
    Flex(Arc<Qwen3TtsModelInner<burn::backend::Flex>>),
    #[cfg(feature = "wgpu")]
    Wgpu(Arc<Qwen3TtsModelInner<burn::backend::Wgpu>>),
    // ...
}
```

The enum is the backend-erased resident model handle. Each variant directly
holds `Arc<ModelInner<B>>`; no extra backend handle wrapper is introduced in v1.

### Session Objects

```rust
struct SessionImpl<B: Backend> {
    inner: Arc<Qwen3TtsModelInner<B>>,
    run: TalkerGenerator<B>,
    session_id: usize,
}

enum Qwen3TtsSession {
    #[cfg(feature = "flex")]
    Flex(SessionImpl<burn::backend::Flex>),
    #[cfg(feature = "wgpu")]
    Wgpu(SessionImpl<burn::backend::Wgpu>),
    // ...
}
```

Session state stays deliberately lean:

- `inner`
- mutable generator state
- session id

It does not store original request, run options, or pre-lowering artifacts.

## Request Compiler

```rust
struct Qwen3TtsRequestCompiler {
    tokenizer: Tokenizer,
    profiles: Qwen3TtsCompiledProfiles,
}

struct Qwen3TtsCompiledProfiles {
    base: Option<Qwen3TtsCompiledProfile>,
    custom_voice: Option<Qwen3TtsCompiledProfile>,
}

struct Qwen3TtsCompiledProfile {
    generation_config: GenerationConfig,
    control_config: Qwen3TtsControlConfig,
    prompt_recipe: Qwen3TtsPromptRecipe,
}

enum Qwen3TtsPromptRecipe {
    Base,
    CustomVoice,
}
```

Rules:

- profile config is loaded once at engine load time
- profile container uses fixed fields, not a dynamic map
- prompt behavior is part of the compiled profile, not split back out into a
  separate public abstraction

## Single Session Seed

The old multi-stage request-preparation graph is replaced by one backend-specific
session-start artifact.

```rust
struct SessionSeed<B: Backend> {
    inputs_embeds: Tensor<B, 3>,
    position_ids: Tensor<B, 3, Int>,
    attention_mask: Tensor<B, 2, Int>,
    trailing_text_hidden: Tensor<B, 3>,
    tts_pad_embed: Tensor<B, 3>,
    codec_eos_token_id: usize,
    suppress_token_ids: Vec<usize>,
}
```

`codec_eos_token_id` is mandatory in the new contract. Missing EOS semantics
must fail before runtime session execution.

## `ModelInner` Internal Methods

```rust
impl<B: Backend> Qwen3TtsModelInner<B> {
    fn compile_session_seed(
        &self,
        request: QwenRequest,
    ) -> Result<SessionSeed<B>, Qwen3TtsError>;

    fn start_generator(
        &self,
        seed: SessionSeed<B>,
        options: Qwen3TtsRunOptions,
    ) -> Result<TalkerGenerator<B>, Qwen3TtsError>;

    fn finalize_audio(
        &self,
        run: &TalkerGenerator<B>,
    ) -> Result<tts_infer::PcmAudio, Qwen3TtsError>;
}
```

Meaning:

- `compile_session_seed()` closes request validation + lowering inside model
  logic
- `start_generator()` consumes the compiled seed and run options
- `finalize_audio()` converts terminal generator state into final PCM output
