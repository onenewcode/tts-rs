use crate::runtime::sampling::SamplingConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeBackend {
    Flex,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    WebGpu,
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub max_new_tokens: usize,
    pub chunk_steps: usize,
    pub sampling: SamplingConfig,
    pub stream: bool,
    pub profiling: bool,
    pub backend: Option<ComputeBackend>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            chunk_steps: 8,
            sampling: SamplingConfig::greedy(),
            stream: false,
            profiling: false,
            backend: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub waveform_pcm: Vec<i16>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub pcm: Vec<i16>,
    pub sample_rate: u32,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
pub enum SynthesisEvent {
    CodecChunk { steps: usize },
    AudioChunk(AudioChunk),
    Finished,
}
