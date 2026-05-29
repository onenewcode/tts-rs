use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::model::graph::engine::components::generator::graph::model::Qwen3TtsCheckpoint;
use crate::model::graph::engine::components::generator::import::config::Qwen3TtsConfig;
use crate::model::graph::engine::components::generator::import::remap::talker_load_key_remapper;

#[derive(Debug)]
pub struct LoadedQwen3TtsTalker<B: Backend> {
    pub config: Qwen3TtsConfig,
    pub model: Qwen3TtsCheckpoint<B>,
}

pub fn load_qwen3_tts_talker_for_inference<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    load_qwen3_tts_talker_with_adapter::<B, _>(
        config_path,
        weights_path,
        device,
        PyTorchToBurnAdapter,
    )
}

fn load_qwen3_tts_talker_with_adapter<B: Backend, A: ModuleAdapter + 'static>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
    adapter: A,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts talker"
    );
    let config = Qwen3TtsConfig::load_from_path(&config_path)?;
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

    let applied = apply_result.applied.len();
    let skipped = apply_result.skipped.len();
    let missing = apply_result.missing.len();
    let unused = apply_result.unused.len();
    if unused != 0 {
        tracing::warn!(
            unused,
            "qwen3 tts talker weights left tensors unused during load"
        );
    }
    tracing::info!(
        applied,
        skipped,
        missing,
        unused,
        "loaded qwen3 tts talker weights"
    );

    Ok(LoadedQwen3TtsTalker { config, model })
}
