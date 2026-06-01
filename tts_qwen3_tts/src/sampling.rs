#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: None,
            repetition_penalty: None,
        }
    }
}
// TODO 完全不正确，你应该加载模型的配置或者默认，而不是自己生成
impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            ..Default::default()
        }
    }
}
