pub mod audio_codec {
    pub mod decoder;
    pub mod encoder;
    pub mod wave_decoder;
}
mod audio_codec_build_decoder;
mod audio_codec_build_encoder;
mod build;
pub mod config;
pub mod load;
pub mod qwen_tts;
pub(crate) mod variant;
