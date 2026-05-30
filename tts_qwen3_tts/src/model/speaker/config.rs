use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SpeakerEncoderConfigManifest {
    #[serde(default = "default_mel_dim")]
    pub(crate) mel_dim: usize,
    #[serde(default = "default_enc_dim")]
    pub(crate) enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub(crate) enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub(crate) enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub(crate) enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub(crate) enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub(crate) enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub(crate) enc_se_channels: usize,
    #[serde(default = "default_speaker_sample_rate")]
    pub(crate) sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelConfigWithSpeaker {
    #[serde(default)]
    pub(crate) speaker_encoder_config: Option<SpeakerEncoderConfigManifest>,
}

fn default_mel_dim() -> usize {
    128
}

fn default_enc_dim() -> usize {
    1024
}

fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}

fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}

fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}

fn default_enc_attention_channels() -> usize {
    128
}

fn default_enc_res2net_scale() -> usize {
    8
}

fn default_enc_se_channels() -> usize {
    128
}

fn default_speaker_sample_rate() -> u32 {
    24_000
}
