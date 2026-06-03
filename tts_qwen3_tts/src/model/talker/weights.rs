use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::Qwen3TtsLoadError;
use crate::model::talker::config::Qwen3TtsTalkerConfig;
use crate::model::talker::network::Qwen3TtsTalkerModelBundle;

const TALKER_LOAD_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.weight$", "${1}norm.gamma")];
#[cfg(test)]
const TALKER_EXPORT_KEY_PATTERNS: [(&str, &str); 1] = [(r"(.*)norm\.gamma$", "${1}norm.weight")];

#[derive(Debug)]
pub struct LoadedQwen3TtsTalker<B: Backend> {
    pub config: Qwen3TtsTalkerConfig,
    pub model: Qwen3TtsTalkerModelBundle<B>,
}

pub fn load_qwen3_tts_talker_for_inference<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3TtsTalker<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        "loading qwen3 tts talker"
    );
    let config = Qwen3TtsTalkerConfig::load_from_path(&config_path)?;
    let mut model = config.init_model_bundle(device);

    let mut store = SafetensorsStore::from_file(&weights_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(
            KeyRemapper::from_patterns(TALKER_LOAD_KEY_PATTERNS.to_vec())
                .expect("static regex remapping must compile"),
        )
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
#[cfg(test)]
mod tests {
    use burn_store::KeyRemapper;

    use super::{TALKER_EXPORT_KEY_PATTERNS, TALKER_LOAD_KEY_PATTERNS};

    #[test]
    fn talker_remappers_compile() {
        let _ = KeyRemapper::from_patterns(TALKER_LOAD_KEY_PATTERNS.to_vec())
            .expect("static load regex remapping must compile");
        let _ = KeyRemapper::from_patterns(TALKER_EXPORT_KEY_PATTERNS.to_vec())
            .expect("static export regex remapping must compile");
    }
}
