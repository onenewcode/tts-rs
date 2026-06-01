mod config;

pub use config::Qwen3TtsProfilingConfig;
// TODO 该方法真的需靠吗
pub(crate) fn configure(config: &Qwen3TtsProfilingConfig) {
    let _ = config;
}
