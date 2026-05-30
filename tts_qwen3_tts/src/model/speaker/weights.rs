use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use super::config::ModelConfigWithSpeaker;
use super::feature::MelSpectrogram;
use super::network::SpeakerEncoderNetwork;
use crate::Qwen3TtsLoadError;

#[derive(Debug)]
pub(crate) struct LoadedQwen3TtsSpeakerEncoder<B: Backend> {
    pub(crate) encoder: SpeakerEncoderNetwork<B>,
    pub(crate) mel_extractor: MelSpectrogram,
    sample_rate: u32,
    pub(crate) device: B::Device,
}

impl<B> LoadedQwen3TtsSpeakerEncoder<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn load(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Option<Self>, Qwen3TtsLoadError> {
        let config_path = config_path.as_ref().to_path_buf();
        let weights_path = weights_path.as_ref().to_path_buf();
        let raw = std::fs::read_to_string(&config_path).map_err(|source| {
            Qwen3TtsLoadError::CompilerConfigIo {
                path: config_path.clone(),
                source,
            }
        })?;
        let config: ModelConfigWithSpeaker = serde_json::from_str(&raw).map_err(|source| {
            Qwen3TtsLoadError::CompilerConfigParse {
                path: config_path.clone(),
                source,
            }
        })?;
        let Some(speaker_config) = config.speaker_encoder_config else {
            return Ok(None);
        };

        let mut encoder = SpeakerEncoderNetwork::new(&speaker_config, device);
        let remapper = KeyRemapper::from_patterns(vec![(r"^speaker_encoder\.(.*)$", "${1}")])
            .expect("static speaker encoder remapping must compile");
        let mut store = SafetensorsStore::from_file(&weights_path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .remap(remapper)
            .skip_enum_variants(true);
        let apply_result =
            encoder
                .load_from(&mut store)
                .map_err(|source| Qwen3TtsLoadError::Store {
                    path: weights_path.clone(),
                    source,
                })?;
        if apply_result.applied.is_empty() {
            return Ok(None);
        }

        tracing::info!(
            applied = apply_result.applied.len(),
            missing = apply_result.missing.len(),
            unused = apply_result.unused.len(),
            "loaded qwen3 tts speaker encoder weights"
        );

        Ok(Some(Self {
            encoder,
            mel_extractor: MelSpectrogram::new(MelSpectrogram::speaker_encoder()),
            sample_rate: speaker_config.sample_rate,
            device: device.clone(),
        }))
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
