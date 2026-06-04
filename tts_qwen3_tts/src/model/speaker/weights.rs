use std::path::Path;

use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

use super::config::SpeakerConfigEnvelope;
use super::infer::feature::MelSpectrogram;
use super::network::SpeakerEncoderNetwork;
use crate::Qwen3TtsLoadError;
use crate::loading::load_adapter::LoadTimeFloatAdapter;

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
    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

pub(crate) fn load_qwen3_tts_speaker_encoder<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
    tensor_dtype: DType,
) -> Result<LoadedQwen3TtsSpeakerEncoder<B>, Qwen3TtsLoadError>
where
    B::Device: Clone,
{
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    let raw = std::fs::read_to_string(&config_path).map_err(|source| {
        Qwen3TtsLoadError::CompilerConfigIo {
            path: config_path.clone(),
            source,
        }
    })?;
    let config: SpeakerConfigEnvelope =
        serde_json::from_str(&raw).map_err(|source| Qwen3TtsLoadError::CompilerConfigParse {
            path: config_path.clone(),
            source,
        })?;
    let speaker_config =
        config
            .speaker_encoder_config
            .ok_or_else(|| Qwen3TtsLoadError::InvalidModelDir {
                message: format!(
                    "{} does not declare speaker_encoder_config",
                    config_path.display()
                ),
            })?;

    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        dtype = %tensor_dtype.name(),
        "loading qwen3 tts speaker encoder"
    );

    let mut encoder = SpeakerEncoderNetwork::new(&speaker_config, device);
    let remapper = KeyRemapper::from_patterns(vec![(r"^speaker_encoder\.(.*)$", "${1}")])
        .expect("static speaker encoder remapping must compile");
    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter.chain(LoadTimeFloatAdapter::new(tensor_dtype)))
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
        return Err(Qwen3TtsLoadError::InvalidModelDir {
            message: format!(
                "{} did not contain any speaker encoder tensors",
                weights_path.display()
            ),
        });
    }

    tracing::info!(
        applied = apply_result.applied.len(),
        missing = apply_result.missing.len(),
        unused = apply_result.unused.len(),
        "loaded qwen3 tts speaker encoder weights"
    );

    Ok(LoadedQwen3TtsSpeakerEncoder {
        encoder,
        mel_extractor: MelSpectrogram::from_speaker_encoder_config(&speaker_config),
        sample_rate: speaker_config.sample_rate,
        device: device.clone(),
    })
}
