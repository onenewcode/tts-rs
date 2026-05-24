use burn::module::Module;
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct Qwen3TtsCheckpoint<B: Backend> {
    pub talker: Qwen3TtsTalkerForConditionalGeneration<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerForConditionalGeneration<B: Backend> {
    pub model: Qwen3TtsTalkerModel<B>,
    pub text_projection: Qwen3TtsTalkerResizeMlp<B>,
    pub codec_head: Linear<B>,
    pub code_predictor: Qwen3TtsTalkerCodePredictorForConditionalGeneration<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerModel<B: Backend> {
    pub codec_embedding: Embedding<B>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub text_embedding: Embedding<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictorForConditionalGeneration<B: Backend> {
    pub model: Qwen3TtsTalkerCodePredictorModel<B>,
    pub lm_head: Vec<Linear<B>>,
    pub small_to_mtp_projection: Option<Linear<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerCodePredictorModel<B: Backend> {
    pub codec_embedding: Vec<Embedding<B>>,
    pub layers: Vec<Qwen3TtsDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsDecoderLayer<B: Backend> {
    pub self_attn: Qwen3TtsAttention<B>,
    pub mlp: Qwen3TtsTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTextMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerResizeMlp<B: Backend> {
    pub linear_fc1: Linear<B>,
    pub linear_fc2: Linear<B>,
}
