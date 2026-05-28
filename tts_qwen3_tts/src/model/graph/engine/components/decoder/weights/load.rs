use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::model::graph::engine::components::decoder::graph::audio_codec::decoder::Qwen3TtsAudioCodecCheckpoint;
use crate::model::graph::engine::components::decoder::import::config::Qwen3TtsAudioCodecConfig;
use crate::model::graph::engine::components::decoder::import::remap::audio_codec_load_key_remapper;

#[derive(Debug)]
pub struct LoadedQwen3TtsAudioCodec<B: Backend> {
    pub config: Qwen3TtsAudioCodecConfig,
    pub model: Qwen3TtsAudioCodecCheckpoint<B>,
}

pub fn load_qwen3_tts_audio_codec<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsAudioCodec<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts audio codec"
    );
    let config = Qwen3TtsAudioCodecConfig::load_from_path(&config_path)?;
    let mut model = config.init_checkpoint(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(audio_codec_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    let applied = apply_result.applied.len();
    let skipped = apply_result.skipped.len();
    let missing = apply_result.missing.len();
    let unused = apply_result.unused.len();
    tracing::info!(
        applied,
        skipped,
        missing,
        unused,
        "loaded qwen3 tts audio codec weights"
    );

    Ok(LoadedQwen3TtsAudioCodec { config, model })
}
