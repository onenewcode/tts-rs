use std::path::Path;

use tts_core::ModelRegistry;

use crate::runtime::executor::register_qwen_family_model_impl;

pub fn register_qwen_family_model(
    registry: &mut ModelRegistry,
    model_id: impl Into<String>,
    model_dir: impl AsRef<Path>,
    variant: impl AsRef<str>,
) -> bool {
    register_qwen_family_model_impl(registry, model_id, model_dir, variant).unwrap_or(false)
}
