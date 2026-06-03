pub(crate) mod codebook;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod wave;

use burn::module::Module;
use burn::tensor::backend::Backend;

pub(crate) mod activation;
pub(crate) mod conv;
pub(crate) use self::decoder::Qwen3TtsAudioCodecDecoder;
pub(crate) use self::encoder::Qwen3TtsAudioCodecEncoder;

#[derive(Module, Debug)]
pub struct Qwen3TtsAudioCodec<B: Backend> {
    pub decoder: Qwen3TtsAudioCodecDecoder<B>,
    pub encoder: Qwen3TtsAudioCodecEncoder<B>,
}
