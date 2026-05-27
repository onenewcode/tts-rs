use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::model::config::talker::Qwen3TtsConfig;
use crate::model::load_report::LoadReport;
use crate::model::load::talker_remap::talker_load_key_remapper;
use crate::model::qwen_tts::Qwen3TtsCheckpoint;

#[derive(Debug)]
pub struct LoadedQwen3TtsTalker<B: Backend> {
    pub config: Qwen3TtsConfig,
    pub model: Qwen3TtsCheckpoint<B>,
    pub load_report: LoadReport,
}

pub fn load_qwen3_tts_talker_for_inference<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    load_qwen3_tts_talker_with_adapter::<B, _>(model_dir, device, PyTorchToBurnAdapter)
}

fn load_qwen3_tts_talker_with_adapter<B: Backend, A: ModuleAdapter + 'static>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
    adapter: A,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    let model_dir = model_dir.as_ref().to_path_buf();
    let weights_path = model_dir.join("model.safetensors");
    tracing::info!(
        model_dir = %model_dir.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts talker"
    );
    let config = Qwen3TtsConfig::load_from_model_dir(&model_dir)?;
    let mut model = config.init_checkpoint(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(adapter)
        .remap(talker_load_key_remapper())
        .skip_enum_variants(true);

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    if !apply_result.unused.is_empty() {
        return Err(Qwen3TtsLoadError::UnusedTensors {
            unused: apply_result.unused.len(),
        });
    }

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
        "loaded qwen3 tts talker weights"
    );

    Ok(LoadedQwen3TtsTalker {
        config,
        model,
        load_report,
    })
}
