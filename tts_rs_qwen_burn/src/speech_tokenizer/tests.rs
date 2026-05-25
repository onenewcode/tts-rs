use burn::backend::NdArray;

use super::config::{
    Qwen3TtsSpeechTokenizerConfig, Qwen3TtsSpeechTokenizerDecoderConfig,
    Qwen3TtsSpeechTokenizerEncoderConfig,
};
use super::init::common::tensor_param_dims;
use super::init::encoder::{derive_encoder_downsample_factor, derive_encoder_downsample_kernel};
use super::model::encoder::Qwen3TtsSpeechTokenizerEncoderBackboneLayer;
use super::model::wave_decoder::Qwen3TtsSpeechTokenizerWaveDecoderEntry;
use super::remap::{speech_tokenizer_export_key_remapper, speech_tokenizer_load_key_remapper};

type TestBackend = NdArray;

fn sample_config() -> Qwen3TtsSpeechTokenizerConfig {
    Qwen3TtsSpeechTokenizerConfig {
        architectures: vec!["Qwen3TtsSpeechTokenizer".to_string()],
        model_type: "qwen3_tts_tokenizer".to_string(),
        encoder_valid_num_quantizers: 2,
        input_sample_rate: 192,
        output_sample_rate: 24000,
        decode_upsample_rate: 2,
        encode_downsample_rate: 16,
        encoder_config: Qwen3TtsSpeechTokenizerEncoderConfig {
            _frame_rate: 12.0,
            attention_bias: true,
            attention_dropout: 0.0,
            audio_channels: 1,
            codebook_dim: 8,
            codebook_size: 16,
            compress: 2,
            dilation_growth_rate: 2,
            dtype: "float32".to_string(),
            head_dim: 4,
            hidden_act: "silu".to_string(),
            hidden_size: 32,
            initializer_range: 0.02,
            intermediate_size: 64,
            kernel_size: 3,
            last_kernel_size: 3,
            layer_scale_initial_scale: 1e-6,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            normalize: false,
            num_attention_heads: 4,
            num_filters: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            num_quantizers: 6,
            num_residual_layers: 1,
            num_semantic_quantizers: 2,
            pad_mode: "constant".to_string(),
            residual_kernel_size: 3,
            rope_theta: 10000.0,
            sampling_rate: 192,
            sliding_window: 0,
            transformers_version: "0".to_string(),
            trim_right_ratio: 1.0,
            upsample_groups: 1,
            upsampling_ratios: vec![2, 2, 2, 2],
            use_cache: false,
            use_causal_conv: true,
            use_conv_shortcut: false,
            use_streaming: false,
            vector_quantization_hidden_dimension: 8,
        },
        decoder_config: Qwen3TtsSpeechTokenizerDecoderConfig {
            attention_bias: true,
            attention_dropout: 0.0,
            latent_dim: 16,
            codebook_dim: 8,
            codebook_size: 16,
            decoder_dim: 64,
            hidden_act: "silu".to_string(),
            hidden_size: 32,
            intermediate_size: 64,
            layer_scale_initial_scale: 1e-6,
            max_position_embeddings: 128,
            head_dim: 4,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            num_quantizers: 6,
            num_semantic_quantizers: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            semantic_codebook_size: 16,
            sliding_window: 0,
            upsample_rates: vec![2, 2, 2, 2],
            upsampling_ratios: vec![2, 2],
            vector_quantization_hidden_dimension: 8,
        },
        transformers_version: "0".to_string(),
    }
}

fn apply_remapper(remapper: &burn_store::KeyRemapper, key: &str) -> String {
    let mut out = key.to_string();
    for (pattern, replacement) in &remapper.patterns {
        if pattern.is_match(&out) {
            out = pattern.replace_all(&out, replacement.as_str()).to_string();
        }
    }
    out
}

#[test]
fn init_checkpoint_builds_expected_decoder_and_encoder_layouts() {
    let config = sample_config();
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);

    assert_eq!(checkpoint.decoder.pre_transformer.layers.len(), 2);
    assert_eq!(checkpoint.decoder.upsample.len(), 2);
    assert_eq!(checkpoint.decoder.decoder.len(), 7);
    assert!(matches!(
        checkpoint.decoder.decoder.first(),
        Some(Qwen3TtsSpeechTokenizerWaveDecoderEntry::InputConv(_))
    ));
    assert!(matches!(
        checkpoint.decoder.decoder.last(),
        Some(Qwen3TtsSpeechTokenizerWaveDecoderEntry::OutputConv(_))
    ));
    assert_eq!(checkpoint.decoder.quantizer.rvq_first.vq.layers.len(), 1);
    assert_eq!(checkpoint.decoder.quantizer.rvq_rest.vq.layers.len(), 4);
    assert_eq!(checkpoint.encoder.encoder.layers.len(), 15);
    assert!(matches!(
        checkpoint.encoder.encoder.layers[0],
        Qwen3TtsSpeechTokenizerEncoderBackboneLayer::InputConv(_)
    ));
    assert!(matches!(
        checkpoint.encoder.encoder.layers[1],
        Qwen3TtsSpeechTokenizerEncoderBackboneLayer::Resnet(_)
    ));
    assert!(matches!(
        checkpoint.encoder.encoder.layers[2],
        Qwen3TtsSpeechTokenizerEncoderBackboneLayer::Empty(_)
    ));
    assert!(matches!(
        checkpoint.encoder.encoder.layers[3],
        Qwen3TtsSpeechTokenizerEncoderBackboneLayer::DownsampleConv(_)
    ));
    assert!(matches!(
        checkpoint.encoder.encoder.layers[14],
        Qwen3TtsSpeechTokenizerEncoderBackboneLayer::OutputConv(_)
    ));
    assert_eq!(
        checkpoint
            .encoder
            .quantizer
            .semantic_residual_vector_quantizer
            .layers
            .len(),
        2
    );
    assert_eq!(
        checkpoint
            .encoder
            .quantizer
            .acoustic_residual_vector_quantizer
            .layers
            .len(),
        4
    );
}

#[test]
fn downsample_helpers_compute_expected_values() {
    let factor = derive_encoder_downsample_factor(192, 16, 192, &[2, 2, 2, 2], 12.0);
    assert_eq!(factor, 1);
    assert_eq!(derive_encoder_downsample_kernel(factor), 2);
}

#[test]
fn helper_initializers_create_expected_param_shapes() {
    let config = sample_config();
    let device = Default::default();
    let checkpoint = config.init_checkpoint::<TestBackend>(&device);
    let first_stage = &checkpoint.decoder.decoder[1];
    let output_activation = &checkpoint.decoder.decoder[5];

    match first_stage {
        Qwen3TtsSpeechTokenizerWaveDecoderEntry::UpsampleStage(stage) => {
            assert_eq!(tensor_param_dims(&stage.block.0.alpha), [64]);
            assert_eq!(stage.block.1.conv.channels, [64, 32]);
        }
        _ => panic!("expected an upsample stage"),
    }

    match output_activation {
        Qwen3TtsSpeechTokenizerWaveDecoderEntry::OutputActivation(activation) => {
            assert_eq!(tensor_param_dims(&activation.alpha), [4]);
            assert_eq!(tensor_param_dims(&activation.beta), [4]);
        }
        _ => panic!("expected output activation"),
    }

    let decoder_codebook = &checkpoint.decoder.quantizer.rvq_first.vq.layers[0]._codebook;
    assert_eq!(tensor_param_dims(&decoder_codebook.cluster_usage), [16]);
    assert_eq!(tensor_param_dims(&decoder_codebook.embedding_sum), [16, 4]);

    let encoder_codebook = &checkpoint
        .encoder
        .quantizer
        .semantic_residual_vector_quantizer
        .layers[0]
        .codebook;
    assert_eq!(tensor_param_dims(&encoder_codebook.initialized), [1]);
    assert_eq!(tensor_param_dims(&encoder_codebook.cluster_usage), [16]);
    assert_eq!(tensor_param_dims(&encoder_codebook.embed_sum), [16, 8]);
}

#[test]
fn speech_tokenizer_load_remapper_only_touches_decoder_rmsnorm_keys() {
    let remapped = apply_remapper(
        &speech_tokenizer_load_key_remapper(),
        "decoder.pre_transformer.layers.0.input_layernorm.weight",
    );
    assert_eq!(
        remapped,
        "decoder.pre_transformer.layers.0.input_layernorm.gamma"
    );

    let untouched = apply_remapper(
        &speech_tokenizer_load_key_remapper(),
        "encoder.encoder_transformer.layers.0.input_layernorm.weight",
    );
    assert_eq!(
        untouched,
        "encoder.encoder_transformer.layers.0.input_layernorm.weight"
    );
}

#[test]
fn speech_tokenizer_export_remapper_reverses_decoder_rmsnorm_keys() {
    let remapped = apply_remapper(
        &speech_tokenizer_export_key_remapper(),
        "decoder.pre_transformer.norm.gamma",
    );
    assert_eq!(remapped, "decoder.pre_transformer.norm.weight");
}
