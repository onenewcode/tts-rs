use std::path::Path;

use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

use crate::Qwen3TtsLoadError;
use crate::loading::load_adapter::LoadTimeFloatAdapter;
use crate::model::codec::config::Qwen3TtsAudioCodecConfig;
use crate::model::codec::network::Qwen3TtsAudioCodec;

const SPEECH_TOKENIZER_LOAD_KEY_PATTERNS: [(&str, &str); 3] = [
    (
        r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.weight$",
        "${1}.gamma",
    ),
    (
        r"^encoder\.encoder\.layers\.(\d+)\.block\.1\.conv\.(weight|bias)$",
        "encoder.encoder.layers.${1}.conv_in.conv.${2}",
    ),
    (
        r"^encoder\.encoder\.layers\.(\d+)\.block\.3\.conv\.(weight|bias)$",
        "encoder.encoder.layers.${1}.conv_out.conv.${2}",
    ),
];
#[cfg(test)]
const SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS: [(&str, &str); 3] = [
    (
        r"^(decoder\.pre_transformer(?:\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|\.norm))\.gamma$",
        "${1}.weight",
    ),
    (
        r"^encoder\.encoder\.layers\.(\d+)\.conv_in\.conv\.(weight|bias)$",
        "encoder.encoder.layers.${1}.block.1.conv.${2}",
    ),
    (
        r"^encoder\.encoder\.layers\.(\d+)\.conv_out\.conv\.(weight|bias)$",
        "encoder.encoder.layers.${1}.block.3.conv.${2}",
    ),
];

#[derive(Debug)]
pub struct LoadedQwen3TtsAudioCodec<B: Backend> {
    pub config: Qwen3TtsAudioCodecConfig,
    pub model: Qwen3TtsAudioCodec<B>,
}

pub fn load_qwen3_tts_audio_codec<B: Backend>(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
    tensor_dtype: Option<DType>,
) -> Result<LoadedQwen3TtsAudioCodec<B>, Qwen3TtsLoadError> {
    let config_path = config_path.as_ref().to_path_buf();
    let weights_path = weights_path.as_ref().to_path_buf();
    tracing::info!(
        config_path = %config_path.display(),
        weights_path = %weights_path.display(),
        dtype = %tensor_dtype.map(|dtype| dtype.name()).unwrap_or("original"),
        "loading qwen3 tts audio codec"
    );
    let config = Qwen3TtsAudioCodecConfig::load_from_path(&config_path)?;
    let mut model = config.init_model(device);

    let mut store = {
        let store = SafetensorsStore::from_file(&weights_path).remap(
            KeyRemapper::from_patterns(SPEECH_TOKENIZER_LOAD_KEY_PATTERNS.to_vec())
                .expect("static regex remapping must compile"),
        );
        let store = if let Some(tensor_dtype) = tensor_dtype {
            store.with_from_adapter(
                PyTorchToBurnAdapter.chain(LoadTimeFloatAdapter::new(tensor_dtype)),
            )
        } else {
            store.with_from_adapter(PyTorchToBurnAdapter)
        };
        store.skip_enum_variants(true)
    };

    let apply_result = model
        .load_from(&mut store)
        .map_err(|source| Qwen3TtsLoadError::Store {
            path: weights_path.clone(),
            source,
        })?;

    tracing::info!(
        applied = apply_result.applied.len(),
        skipped = apply_result.skipped.len(),
        missing = apply_result.missing.len(),
        unused = apply_result.unused.len(),
        "loaded qwen3 tts audio codec weights"
    );

    Ok(LoadedQwen3TtsAudioCodec { config, model })
}

#[cfg(test)]
mod tests {
    use burn_store::KeyRemapper;

    use super::{SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS, SPEECH_TOKENIZER_LOAD_KEY_PATTERNS};

    #[test]
    fn audio_codec_remappers_compile() {
        let _ = KeyRemapper::from_patterns(SPEECH_TOKENIZER_LOAD_KEY_PATTERNS.to_vec())
            .expect("static load regex remapping must compile");
        let _ = KeyRemapper::from_patterns(SPEECH_TOKENIZER_EXPORT_KEY_PATTERNS.to_vec())
            .expect("static export regex remapping must compile");
    }
}
