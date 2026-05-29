use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::{Tensor, backend::Backend};

use crate::kernels::activation::{AudioCodecLayerScale, Qwen3TtsAudioCodecEmptyModule};
use crate::kernels::conv::AudioCodecCausalConv1d;
/// TODO 实现全部不正确
#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoder<B: Backend> {
    pub encoder: Qwen3TtsAudioCodecEncoderBackbone<B>,
    pub encoder_transformer: Qwen3TtsAudioCodecEncoderTransformer<B>,
    pub downsample: AudioCodecCausalConv1d<B>,
    pub quantizer: Qwen3TtsAudioCodecEncoderQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderBackbone<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecEncoderBackboneLayer<B>>,
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3TtsAudioCodecEncoderBackboneLayer<B: Backend> {
    InputConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
    Resnet(Qwen3TtsAudioCodecEncoderResnetLayer<B>),
    DownsampleConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
    OutputConv(Qwen3TtsAudioCodecEncoderConvLayer<B>),
    Empty(Qwen3TtsAudioCodecEmptyModule),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderConvLayer<B: Backend> {
    pub conv: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderResnetLayer<B: Backend> {
    pub block: (
        Qwen3TtsAudioCodecEmptyModule,
        AudioCodecCausalConv1d<B>,
        Qwen3TtsAudioCodecEmptyModule,
        AudioCodecCausalConv1d<B>,
    ),
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderTransformer<B: Backend> {
    pub layers: Vec<Qwen3TtsAudioCodecEncoderTransformerLayer<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderTransformerLayer<B: Backend> {
    pub self_attn: Qwen3TtsAudioCodecEncoderAttention<B>,
    pub mlp: Qwen3TtsAudioCodecEncoderMlp<B>,
    pub input_layernorm: LayerNorm<B>,
    pub post_attention_layernorm: LayerNorm<B>,
    pub self_attn_layer_scale: AudioCodecLayerScale<B>,
    pub mlp_layer_scale: AudioCodecLayerScale<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderQuantizer<B: Backend> {
    pub semantic_residual_vector_quantizer: Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B>,
    pub acoustic_residual_vector_quantizer: Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B: Backend> {
    pub input_proj: Conv1d<B>,
    pub output_proj: Conv1d<B>,
    pub layers: Vec<Qwen3TtsAudioCodecEncoderVectorQuantization<B>>,
}
/// TODO 为什么要无意义的套一层
#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderVectorQuantization<B: Backend> {
    pub codebook: Qwen3TtsAudioCodecEncoderCodebook<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodecEncoderCodebook<B: Backend> {
    pub initialized: Param<Tensor<B, 1>>,
    pub cluster_usage: Param<Tensor<B, 1>>,
    pub embed_sum: Param<Tensor<B, 2>>,
}
