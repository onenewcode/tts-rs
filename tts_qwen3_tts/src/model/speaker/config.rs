use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct SpeakerEncoderConfig {
    pub(crate) mel_dim: usize,
    pub(crate) enc_dim: usize,
    pub(crate) enc_channels: Vec<usize>,
    pub(crate) enc_kernel_sizes: Vec<usize>,
    pub(crate) enc_dilations: Vec<usize>,
    pub(crate) enc_attention_channels: usize,
    pub(crate) enc_res2net_scale: usize,
    pub(crate) enc_se_channels: usize,
    pub(crate) sample_rate: u32,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
            sample_rate: 24_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SpeakerConfigEnvelope {
    #[serde(default)]
    pub(crate) speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

#[cfg(test)]
mod tests {
    use super::{SpeakerConfigEnvelope, SpeakerEncoderConfig};

    #[test]
    fn speaker_encoder_config_defaults_missing_fields() {
        let config: SpeakerEncoderConfig =
            serde_json::from_str("{}").expect("empty speaker config should deserialize");

        assert_eq!(config.mel_dim, 128);
        assert_eq!(config.enc_dim, 1024);
        assert_eq!(config.enc_channels, vec![512, 512, 512, 512, 1536]);
        assert_eq!(config.enc_kernel_sizes, vec![5, 3, 3, 3, 1]);
        assert_eq!(config.enc_dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(config.enc_attention_channels, 128);
        assert_eq!(config.enc_res2net_scale, 8);
        assert_eq!(config.enc_se_channels, 128);
        assert_eq!(config.sample_rate, 24_000);
    }

    #[test]
    fn speaker_config_envelope_defaults_to_missing_speaker_section() {
        let config: SpeakerConfigEnvelope =
            serde_json::from_str("{}").expect("empty envelope should deserialize");

        assert!(config.speaker_encoder_config.is_none());
    }
}
