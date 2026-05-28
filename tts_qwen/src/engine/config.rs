use crate::profiling::ProfilingConfig;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub profiling: ProfilingConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            profiling: ProfilingConfig::default(),
        }
    }
}
