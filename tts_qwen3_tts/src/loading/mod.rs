pub(crate) mod package;

use crate::capabilities::project_capabilities;
use crate::execution::Qwen3LoadedModelInstance;
use crate::execution::compiler::Qwen3TtsRequestCompiler;
use crate::loading::package::Qwen3TtsPackage;
use crate::model::Qwen3TtsLoadedModel;
use crate::{Qwen3TtsEngineConfig, Qwen3TtsLoadError};

pub(crate) fn load_instance(
    config: &Qwen3TtsEngineConfig,
) -> Result<Qwen3LoadedModelInstance, Qwen3TtsLoadError> {
    let package = Qwen3TtsPackage::load(&config.package)?;
    let compiler = Qwen3TtsRequestCompiler::load(&package)?;
    let model = Qwen3TtsLoadedModel::load(
        package.clone(),
        config.backend,
        &config.profiling,
        compiler.clone(),
    )?;
    let capabilities = project_capabilities(&package, config.backend, &compiler, &model);
    Ok(Qwen3LoadedModelInstance::new(
        model,
        package,
        config.backend,
        config.profiling.clone(),
        capabilities,
    ))
}
