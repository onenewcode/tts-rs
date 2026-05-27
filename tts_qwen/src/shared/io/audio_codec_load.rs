use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::audio_codec::Qwen3TtsAudioCodecCheckpoint;
use crate::shared::config::audio_codec::Qwen3TtsAudioCodecConfig;
use crate::shared::io::LoadReport;
use crate::shared::io::audio_codec_remap::audio_codec_load_key_remapper;

#[derive(Debug)]
pub struct LoadedQwen3TtsAudioCodec<B: Backend> {
    pub config: Qwen3TtsAudioCodecConfig,
    pub model: Qwen3TtsAudioCodecCheckpoint<B>,
    pub load_report: LoadReport,
    pub model_dir: PathBuf,
    pub weights_path: PathBuf,
}

pub fn load_qwen3_tts_audio_codec<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsAudioCodec<B>, Qwen3TtsLoadError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let audio_codec_weights = model_dir.join("audio_codec").join("model.safetensors");
    let speech_tokenizer_weights = model_dir.join("speech_tokenizer").join("model.safetensors");
    let weights_path = if audio_codec_weights.exists() {
        audio_codec_weights
    } else {
        speech_tokenizer_weights
    };
    tracing::info!(
        model_dir = %model_dir.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts audio codec"
    );
    let config = Qwen3TtsAudioCodecConfig::load_from_model_dir(&model_dir)?;
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

    let load_report = LoadReport {
        applied: apply_result.applied.len(),
        skipped: apply_result.skipped.len(),
        missing: apply_result.missing.len(),
        unused: apply_result.unused.len(),
    };
    tracing::info!(
        applied = load_report.applied,
        skipped = load_report.skipped,
        missing = load_report.missing,
        unused = load_report.unused,
        "loaded qwen3 tts audio codec weights"
    );

    Ok(LoadedQwen3TtsAudioCodec {
        config,
        model,
        load_report,
        model_dir,
        weights_path,
    })
}
