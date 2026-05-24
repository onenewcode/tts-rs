use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear, RmsNorm};
use burn::tensor::{Tensor, backend::Backend};

use super::common::{TokenizerCausalConv1d, TokenizerLayerScale};
use super::encoder::Qwen3TtsSpeechTokenizerEncoder;
use super::wave_decoder::Qwen3TtsSpeechTokenizerWaveDecoderEntry;

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerCheckpoint<B: Backend> {
    pub decoder: Qwen3TtsSpeechTokenizerDecoder<B>,
    pub encoder: Qwen3TtsSpeechTokenizerEncoder<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoder<B: Backend> {
    pub pre_transformer: Qwen3TtsSpeechTokenizerDecoderTransformer<B>,
    pub quantizer: Qwen3TtsSpeechTokenizerDecoderQuantizer<B>,
    pub pre_conv: TokenizerCausalConv1d<B>,
    pub upsample: Vec<(
        super::common::TokenizerCausalTransConv1d<B>,
        Qwen3TtsSpeechTokenizerConvNeXtBlock<B>,
    )>,
    pub decoder: Vec<Qwen3TtsSpeechTokenizerWaveDecoderEntry<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerConvNeXtBlock<B: Backend> {
    pub dwconv: TokenizerCausalConv1d<B>,
    pub norm: LayerNorm<B>,
    pub pwconv1: Linear<B>,
    pub pwconv2: Linear<B>,
    pub gamma: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerDecoderTransformerLayer<B>>,
    pub norm: RmsNorm<B>,
    pub input_proj: Linear<B>,
    pub output_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsSpeechTokenizerDecoderAttention<B>,
    pub mlp: Qwen3TtsSpeechTokenizerDecoderMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub self_attn_layer_scale: TokenizerLayerScale<B>,
    pub mlp_layer_scale: TokenizerLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderQuantizer<B: Backend> {
    pub rvq_first: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B>,
    pub rvq_rest: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub vq: Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderResidualVectorQuantization<B: Backend> {
    pub layers: Vec<Qwen3TtsSpeechTokenizerDecoderVectorQuantization<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderVectorQuantization<B: Backend> {
    pub _codebook: Qwen3TtsSpeechTokenizerDecoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderCodebook<B: Backend> {
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embedding_sum: Param<Tensor<B, 2>>,
}
