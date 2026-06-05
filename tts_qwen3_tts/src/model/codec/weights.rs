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

const AUDIO_CODEC_LOAD_KEY_PATTERNS: [(&str, &str); 4] = [
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
    // Legacy checkpoints stored decoder quantizer stats under `._codebook`; keep the
    // module field named `codebook` and remap only at the checkpoint boundary.
    (
        r"^(decoder\.quantizer\.rvq_(?:first|rest)\.vq\.layers\.\d+)\._codebook\.(cluster_usage|embedding_sum)$",
        "${1}.codebook.${2}",
    ),
];
#[cfg(test)]
const AUDIO_CODEC_EXPORT_KEY_PATTERNS: [(&str, &str); 4] = [
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
    (
        r"^(decoder\.quantizer\.rvq_(?:first|rest)\.vq\.layers\.\d+)\.codebook\.(cluster_usage|embedding_sum)$",
        "${1}._codebook.${2}",
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
            KeyRemapper::from_patterns(AUDIO_CODEC_LOAD_KEY_PATTERNS.to_vec())
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
    use burn::module::ParamId;
    use burn::tensor::Tensor;
    use burn_store::{KeyRemapper, TensorSnapshot};

    use crate::loading::runtime::RuntimeBackend;

    use super::{AUDIO_CODEC_EXPORT_KEY_PATTERNS, AUDIO_CODEC_LOAD_KEY_PATTERNS};

    #[test]
    fn audio_codec_remappers_compile() {
        let _ = KeyRemapper::from_patterns(AUDIO_CODEC_LOAD_KEY_PATTERNS.to_vec())
            .expect("static load regex remapping must compile");
        let _ = KeyRemapper::from_patterns(AUDIO_CODEC_EXPORT_KEY_PATTERNS.to_vec())
            .expect("static export regex remapping must compile");
    }

    #[test]
    fn audio_codec_remapper_maps_legacy_decoder_codebook_keys() {
        let remapper = KeyRemapper::from_patterns(AUDIO_CODEC_LOAD_KEY_PATTERNS.to_vec())
            .expect("static load regex remapping must compile");
        let device = Default::default();
        let tensor = Tensor::<RuntimeBackend, 1>::zeros([1], &device);
        let snapshot = TensorSnapshot::from_float(
            &tensor,
            vec![
                "decoder".into(),
                "quantizer".into(),
                "rvq_first".into(),
                "vq".into(),
                "layers".into(),
                "0".into(),
                "_codebook".into(),
                "cluster_usage".into(),
            ],
            Vec::new(),
            ParamId::new(),
        );

        let (snapshots, remapped_names) = remapper.remap(vec![snapshot]);

        assert_eq!(
            snapshots[0].full_path(),
            "decoder.quantizer.rvq_first.vq.layers.0.codebook.cluster_usage"
        );
        assert_eq!(
            remapped_names[0],
            (
                "decoder.quantizer.rvq_first.vq.layers.0.codebook.cluster_usage".to_string(),
                "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage".to_string(),
            )
        );
    }
}
