#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsProfilingConfig {
    pub enabled: bool,
    pub per_step: bool,
    pub stage_summary: bool,
    pub log_topk: usize,
}

impl Default for Qwen3TtsProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            per_step: false,
            stage_summary: true,
            log_topk: 8,
        }
    }
}
