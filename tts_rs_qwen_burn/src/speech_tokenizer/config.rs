use std::path::Path;

use burn::config::Config;

use crate::Qwen3TtsLoadError;

#[derive(Config, Debug)]
pub struct Qwen3TtsSpeechTokenizerConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_config: Qwen3TtsSpeechTokenizerEncoderConfig,
    pub decoder_config: Qwen3TtsSpeechTokenizerDecoderConfig,
    pub transformers_version: String,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsSpeechTokenizerDecoderConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub semantic_codebook_size: usize,
    pub sliding_window: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

#[derive(Config, Debug)]
pub struct Qwen3TtsSpeechTokenizerEncoderConfig {
    pub _frame_rate: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub audio_channels: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub compress: usize,
    pub dilation_growth_rate: usize,
    pub dtype: String,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub normalize: bool,
    pub num_attention_heads: usize,
    pub num_filters: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_residual_layers: usize,
    pub num_semantic_quantizers: usize,
    pub pad_mode: String,
    pub residual_kernel_size: usize,
    pub rope_theta: f64,
    pub sampling_rate: usize,
    pub sliding_window: usize,
    pub transformers_version: String,
    pub trim_right_ratio: f64,
    pub upsample_groups: usize,
    pub upsampling_ratios: Vec<usize>,
    pub use_cache: bool,
    pub use_causal_conv: bool,
    pub use_conv_shortcut: bool,
    pub use_streaming: bool,
    pub vector_quantization_hidden_dimension: usize,
}

impl Qwen3TtsSpeechTokenizerConfig {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3TtsLoadError> {
        let path = model_dir
            .as_ref()
            .join("speech_tokenizer")
            .join("config.json");
        Self::load(&path).map_err(|source| Qwen3TtsLoadError::Config { path, source })
    }
}
