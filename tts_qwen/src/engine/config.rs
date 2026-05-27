use crate::profiling::ProfilingConfig;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_seq_len: usize,
    pub codec_chunk_steps: usize,
    pub max_concurrent_sessions: usize,
    pub profiling: ProfilingConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            codec_chunk_steps: 8,
            max_concurrent_sessions: 1,
            profiling: ProfilingConfig::default(),
        }
    }
}
