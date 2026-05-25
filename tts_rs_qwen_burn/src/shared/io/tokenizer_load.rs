use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::shared::manifest::LoadReport;

use crate::shared::config::tokenizer::Qwen3TtsSpeechTokenizerConfig;
use crate::speech_tokenizer::Qwen3TtsSpeechTokenizerCheckpoint;
use crate::shared::io::tokenizer_remap::speech_tokenizer_load_key_remapper;

#[derive(Debug)]
pub struct LoadedQwen3TtsSpeechTokenizer<B: Backend> {
    pub config: Qwen3TtsSpeechTokenizerConfig,
    pub model: Qwen3TtsSpeechTokenizerCheckpoint<B>,
    pub load_report: LoadReport,
    pub model_dir: PathBuf,
    pub weights_path: PathBuf,
}

pub fn load_qwen3_tts_speech_tokenizer<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsSpeechTokenizer<B>, Qwen3TtsLoadError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let weights_path = model_dir.join("speech_tokenizer").join("model.safetensors");
    let config = Qwen3TtsSpeechTokenizerConfig::load_from_model_dir(&model_dir)?;
    let mut model = config.init_checkpoint(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(speech_tokenizer_load_key_remapper())
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

    Ok(LoadedQwen3TtsSpeechTokenizer {
        config,
        model,
        load_report,
        model_dir,
        weights_path,
    })
}
