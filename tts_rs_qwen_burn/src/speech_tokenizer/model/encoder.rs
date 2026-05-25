use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::{Tensor, backend::Backend};

use crate::shared::nn::activation::{Qwen3TtsSpeechTokenizerEmptyModule, TokenizerLayerScale};
use crate::shared::nn::conv::TokenizerCausalConv1d;

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoder<B: Backend> {
    pub encoder: Qwen3TtsSpeechTokenizerEncoderBackbone<B>,
    pub encoder_transformer: Qwen3TtsSpeechTokenizerEncoderTransformer<B>,
    pub downsample: TokenizerCausalConv1d<B>,
    pub quantizer: Qwen3TtsSpeechTokenizerEncoderQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderBackbone<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderBackboneLayer<B>>,
}

#[derive(Module, Debug)]
pub enum Qwen3TtsSpeechTokenizerEncoderBackboneLayer<B: Backend> {
    InputConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    Resnet(Qwen3TtsSpeechTokenizerEncoderResnetLayer<B>),
    DownsampleConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    OutputConv(Qwen3TtsSpeechTokenizerEncoderConvLayer<B>),
    Empty(Qwen3TtsSpeechTokenizerEmptyModule),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderConvLayer<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderResnetLayer<B: Backend> {
    pub block: (
        Qwen3TtsSpeechTokenizerEmptyModule,
        TokenizerCausalConv1d<B>,
        Qwen3TtsSpeechTokenizerEmptyModule,
        TokenizerCausalConv1d<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderTransformerLayer<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsSpeechTokenizerEncoderAttention<B>,
    pub mlp: Qwen3TtsSpeechTokenizerEncoderMlp<B>,
    pub input_layernorm: LayerNorm<B>,
    pub post_attention_layernorm: LayerNorm<B>,
    pub self_attn_layer_scale: TokenizerLayerScale<B>,
    pub mlp_layer_scale: TokenizerLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderQuantizer<B: Backend> {
    pub semantic_residual_vector_quantizer:
        Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B>,
    pub acoustic_residual_vector_quantizer:
        Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub layers: Vec<Qwen3TtsSpeechTokenizerEncoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderVectorQuantization<B: Backend> {
    pub codebook: Qwen3TtsSpeechTokenizerEncoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderCodebook<B: Backend> {
    pub initialized: Param<Tensor<B, 1>>,
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embed_sum: Param<Tensor<B, 2>>,
}
