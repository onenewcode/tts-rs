use burn::nn::conv::Conv1d;
use burn::nn::{LayerNorm, RotaryEncoding, RotaryEncodingConfig};
use burn::tensor::activation::{elu, gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::{ConvOptions, PadMode};
use burn::tensor::{DType, Tensor, TensorData};

use crate::model::graph::engine::components::decoder::graph::audio_codec::encoder::{
    Qwen3TtsAudioCodecEncoderAttention, Qwen3TtsAudioCodecEncoderBackbone,
    Qwen3TtsAudioCodecEncoderBackboneLayer, Qwen3TtsAudioCodecEncoderQuantizer,
    Qwen3TtsAudioCodecEncoderResidualVectorQuantizer, Qwen3TtsAudioCodecEncoderResnetLayer,
    Qwen3TtsAudioCodecEncoderTransformer, Qwen3TtsAudioCodecEncoderTransformerLayer,
    Qwen3TtsAudioCodecEncoderVectorQuantization,
};
use crate::model::graph::engine::components::decoder::weights::LoadedQwen3TtsAudioCodec;
use crate::model::speaker::LoadedQwen3TtsSpeakerEncoder;
use crate::runtime::reference_audio::load_reference_audio;
use crate::{
    BaseVoiceCloneReferenceAudio, Qwen3TtsInferenceError, Qwen3TtsVoiceClonePrompt,
    Qwen3TtsVoiceClonePromptMode,
};

pub(crate) fn create_voice_clone_prompt<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    speaker_encoder: &LoadedQwen3TtsSpeakerEncoder<B>,
    device: &B::Device,
    reference: &BaseVoiceCloneReferenceAudio,
) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsInferenceError>
where
    B::Device: Clone,
{
    let transcript = reference
        .transcript
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if !reference.x_vector_only && transcript.is_none() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "ref_text is required when x_vector_only is false".to_string(),
        });
    }

    let prepared_for_speaker =
        load_reference_audio(&reference.path, speaker_encoder.sample_rate())?;
    let speaker_embedding = speaker_encoder.encode(&prepared_for_speaker.samples)?;
    if speaker_embedding.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "reference audio {} produced no speaker embedding",
                reference.path.display()
            ),
        });
    }

    let ref_codec_token_ids = if reference.x_vector_only {
        None
    } else {
        Some(encode_reference_codec_frames(
            loaded,
            device,
            &load_reference_audio(&reference.path, loaded.config.input_sample_rate as u32)?.samples,
        )?)
    };

    Ok(Qwen3TtsVoiceClonePrompt {
        speaker_embedding,
        ref_codec_token_ids,
        transcript: transcript.map(ToOwned::to_owned),
        mode: if reference.x_vector_only {
            Qwen3TtsVoiceClonePromptMode::XVectorOnly
        } else {
            Qwen3TtsVoiceClonePromptMode::Icl
        },
    })
}

fn encode_reference_codec_frames<B: Backend>(
    loaded: &LoadedQwen3TtsAudioCodec<B>,
    device: &B::Device,
    samples: &[f32],
) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
    let waveform = Tensor::<B, 3>::from_data(
        TensorData::new(samples.to_vec(), [1, 1, samples.len()]),
        device,
    );
    let encoded = run_encoder_backbone(&loaded.model.encoder.encoder, waveform);
    let transformed =
        run_encoder_transformer(&loaded.model.encoder.encoder_transformer, encoded, loaded);
    let downsampled = streamable_conv1d(
        &loaded.model.encoder.downsample.conv,
        transformed,
        ConvPadMode::Replicate,
    );
    extract_reference_codec_frames(&loaded.model.encoder.quantizer, downsampled, loaded)
}

fn extract_reference_codec_frames<B: Backend>(
    quantizer: &Qwen3TtsAudioCodecEncoderQuantizer<B>,
    hidden: Tensor<B, 3>,
    loaded: &LoadedQwen3TtsAudioCodec<B>,
) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
    let semantic_layers = loaded.config.encoder_config.num_semantic_quantizers;
    let valid_layers = loaded.config.encoder_valid_num_quantizers;
    let acoustic_layers = valid_layers.saturating_sub(semantic_layers);

    let semantic_codes = encode_quantizer_group(
        quantizer.semantic_residual_vector_quantizer.clone(),
        hidden.clone(),
        semantic_layers,
    )?;
    let acoustic_codes = encode_quantizer_group(
        quantizer.acoustic_residual_vector_quantizer.clone(),
        hidden,
        acoustic_layers,
    )?;
    if semantic_codes.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "reference audio produced no semantic codec frames".to_string(),
        });
    }

    let time_steps = semantic_codes[0].len();
    let mut frames = Vec::with_capacity(time_steps);
    for time_index in 0..time_steps {
        let mut frame = Vec::with_capacity(valid_layers);
        for layer_codes in &semantic_codes {
            frame.push(layer_codes[time_index]);
        }
        for layer_codes in &acoustic_codes {
            frame.push(layer_codes[time_index]);
        }
        frames.push(frame);
    }
    Ok(frames)
}

fn encode_quantizer_group<B: Backend>(
    rvq: Qwen3TtsAudioCodecEncoderResidualVectorQuantizer<B>,
    hidden: Tensor<B, 3>,
    max_layers: usize,
) -> Result<Vec<Vec<i64>>, Qwen3TtsInferenceError> {
    if max_layers == 0 {
        return Ok(Vec::new());
    }
    let projected = rvq.input_proj.forward(hidden);
    let mut residual = projected.clone();
    let mut all_codes = Vec::with_capacity(max_layers);
    for layer in rvq.layers.iter().take(max_layers) {
        let (codes, quantized) = nearest_codebook_tokens_and_quantized(layer, residual.clone())?;
        residual = residual - quantized;
        all_codes.push(codes);
    }
    Ok(all_codes)
}

fn run_encoder_backbone<B: Backend>(
    backbone: &Qwen3TtsAudioCodecEncoderBackbone<B>,
    mut hidden: Tensor<B, 3>,
) -> Tensor<B, 3> {
    for layer in &backbone.layers {
        hidden = match layer {
            Qwen3TtsAudioCodecEncoderBackboneLayer::InputConv(layer) => {
                streamable_conv1d(&layer.conv, hidden, ConvPadMode::Constant)
            }
            Qwen3TtsAudioCodecEncoderBackboneLayer::DownsampleConv(layer)
            | Qwen3TtsAudioCodecEncoderBackboneLayer::OutputConv(layer) => {
                streamable_conv1d(&layer.conv, elu(hidden, 1.0), ConvPadMode::Constant)
            }
            Qwen3TtsAudioCodecEncoderBackboneLayer::Resnet(layer) => {
                run_encoder_resnet(layer, hidden)
            }
            Qwen3TtsAudioCodecEncoderBackboneLayer::Empty(_) => hidden,
        };
    }
    hidden
}

#[derive(Debug, Clone, Copy)]
enum ConvPadMode {
    Constant,
    Replicate,
}

fn streamable_conv1d<B: Backend>(
    conv: &Conv1d<B>,
    x: Tensor<B, 3>,
    pad_mode: ConvPadMode,
) -> Tensor<B, 3> {
    let time = x.dims()[2];
    let effective_kernel = (conv.kernel_size - 1) * conv.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(conv.stride);
    let extra_padding =
        extra_padding_for_conv1d(time, effective_kernel, conv.stride, padding_total);
    let x = pad_1d(x, padding_total, extra_padding, pad_mode);
    conv1d(
        x,
        conv.weight.val(),
        conv.bias.as_ref().map(|bias| bias.val()),
        ConvOptions::new([conv.stride], [0], [conv.dilation], conv.groups),
    )
}

fn extra_padding_for_conv1d(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames = (len + padding_total).saturating_sub(kernel_size) as f64 / stride as f64 + 1.0;
    let ideal_len = ((n_frames.ceil() as usize).saturating_sub(1) * stride + kernel_size)
        .saturating_sub(padding_total);
    ideal_len.saturating_sub(len)
}

fn pad_1d<B: Backend>(
    x: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
    mode: ConvPadMode,
) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return x;
    }
    match mode {
        ConvPadMode::Constant => x.pad((pad_left, pad_right, 0, 0), PadMode::Constant(0.0)),
        ConvPadMode::Replicate => replicate_pad_1d(x, pad_left, pad_right),
    }
}

fn replicate_pad_1d<B: Backend>(
    x: Tensor<B, 3>,
    pad_left: usize,
    pad_right: usize,
) -> Tensor<B, 3> {
    let [batch, channels, time] = x.dims();
    let mut segments = Vec::with_capacity(3);
    if pad_left > 0 {
        segments.push(
            x.clone()
                .slice([0..batch, 0..channels, 0..1])
                .repeat_dim(2, pad_left),
        );
    }
    segments.push(x.clone());
    if pad_right > 0 {
        segments.push(
            x.slice([0..batch, 0..channels, time - 1..time])
                .repeat_dim(2, pad_right),
        );
    }
    Tensor::cat(segments, 2)
}

fn run_encoder_resnet<B: Backend>(
    layer: &Qwen3TtsAudioCodecEncoderResnetLayer<B>,
    hidden: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let residual = hidden.clone();
    let hidden = layer.block.1.forward(elu(hidden, 1.0));
    let hidden = elu(hidden, 1.0);
    let hidden = layer.block.3.forward(hidden);
    residual + hidden
}

fn run_encoder_transformer<B: Backend>(
    transformer: &Qwen3TtsAudioCodecEncoderTransformer<B>,
    hidden: Tensor<B, 3>,
    loaded: &LoadedQwen3TtsAudioCodec<B>,
) -> Tensor<B, 3> {
    let config = &loaded.config.encoder_config;
    let rope = RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim)
        .with_theta(config.rope_theta as f32)
        .init(&hidden.device());
    let mut hidden = hidden.swap_dims(1, 2);
    for layer in &transformer.layers {
        hidden = run_encoder_transformer_layer(
            layer,
            hidden,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            &rope,
        );
    }
    hidden.swap_dims(1, 2)
}

fn run_encoder_transformer_layer<B: Backend>(
    layer: &Qwen3TtsAudioCodecEncoderTransformerLayer<B>,
    hidden: Tensor<B, 3>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: &RotaryEncoding<B>,
) -> Tensor<B, 3> {
    let residual = hidden.clone();
    let hidden = layer_norm_3d(&layer.input_layernorm, hidden);
    let hidden = run_encoder_attention(
        &layer.self_attn,
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        rope,
    );
    let hidden = layer.self_attn_layer_scale.forward(hidden);
    let hidden = residual + hidden;

    let residual = hidden.clone();
    let hidden = layer_norm_3d(&layer.post_attention_layernorm, hidden);
    let hidden = run_encoder_mlp(&layer.mlp, hidden);
    let hidden = layer.mlp_layer_scale.forward(hidden);
    residual + hidden
}

fn run_encoder_attention<B: Backend>(
    attention: &Qwen3TtsAudioCodecEncoderAttention<B>,
    hidden: Tensor<B, 3>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: &RotaryEncoding<B>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    let device = hidden.device();
    let hidden_2d = hidden.reshape([batch_size * seq_len, hidden_size]);
    let query = attention
        .q_proj
        .forward(hidden_2d.clone())
        .reshape([batch_size, seq_len, num_heads, head_dim])
        .swap_dims(1, 2);
    let key = attention
        .k_proj
        .forward(hidden_2d.clone())
        .reshape([batch_size, seq_len, num_kv_heads, head_dim])
        .swap_dims(1, 2);
    let value = attention
        .v_proj
        .forward(hidden_2d)
        .reshape([batch_size, seq_len, num_kv_heads, head_dim])
        .swap_dims(1, 2);

    let query = rope.apply(query, 0);
    let key = rope.apply(key, 0);
    let key = repeat_kv(key, num_heads / num_kv_heads);
    let value = repeat_kv(value, num_heads / num_kv_heads);

    let dtype = query.dtype();
    let mut attention_scores = query
        .matmul(key.swap_dims(2, 3))
        .div_scalar((head_dim as f32).sqrt());
    attention_scores =
        attention_scores + causal_attention_bias(batch_size, num_heads, seq_len, dtype, &device);
    let attention_weights = softmax(attention_scores.cast(DType::F32), 3).cast(dtype);
    let attention_output = attention_weights.matmul(value);
    let attention_output =
        attention_output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, num_heads * head_dim]);

    attention
        .o_proj
        .forward(attention_output.reshape([batch_size * seq_len, num_heads * head_dim]))
        .reshape([batch_size, seq_len, hidden_size])
}

fn causal_attention_bias<B: Backend>(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for query_idx in 0..seq_len {
        for key_idx in 0..seq_len {
            values.push(if key_idx > query_idx {
                f32::NEG_INFINITY
            } else {
                0.0
            });
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(values, [1, 1, seq_len, seq_len]), device)
        .repeat_dim(0, batch_size)
        .repeat_dim(1, num_heads)
        .cast(dtype)
}

fn run_encoder_mlp<B: Backend>(
    mlp: &crate::model::graph::engine::components::decoder::graph::audio_codec::encoder::Qwen3TtsAudioCodecEncoderMlp<B>,
    hidden: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    let hidden_2d = hidden.reshape([batch_size * seq_len, hidden_size]);
    let hidden = mlp.fc1.forward(hidden_2d);
    let hidden = gelu(hidden);
    mlp.fc2
        .forward(hidden)
        .reshape([batch_size, seq_len, hidden_size])
}

fn layer_norm_3d<B: Backend>(norm: &LayerNorm<B>, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = hidden.dims();
    norm.forward(hidden.reshape([batch_size * seq_len, hidden_size]))
        .reshape([batch_size, seq_len, hidden_size])
}

fn repeat_kv<B: Backend>(hidden: Tensor<B, 4>, repetitions: usize) -> Tensor<B, 4> {
    if repetitions == 1 {
        return hidden;
    }
    let [batch_size, num_heads, seq_len, head_dim] = hidden.dims();
    hidden
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repetitions)
        .reshape([batch_size, num_heads * repetitions, seq_len, head_dim])
}

fn nearest_codebook_tokens_and_quantized<B: Backend>(
    layer: &Qwen3TtsAudioCodecEncoderVectorQuantization<B>,
    hidden: Tensor<B, 3>,
) -> Result<(Vec<i64>, Tensor<B, 3>), Qwen3TtsInferenceError> {
    let [batch_size, hidden_size, time_steps] = hidden.dims();
    if batch_size != 1 || hidden_size == 0 || time_steps == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "semantic quantizer expects [1, hidden, time] with non-zero dims, got [{batch_size}, {hidden_size}, {time_steps}]"
            ),
        });
    }

    let hidden_values = hidden
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|source| Qwen3TtsInferenceError::TensorRead {
            message: format!("failed to read semantic encoder activations: {source}"),
        })?;
    let cluster_usage = layer
        .codebook
        .cluster_usage
        .val()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|source| Qwen3TtsInferenceError::TensorRead {
            message: format!("failed to read semantic codebook usage: {source}"),
        })?;
    let embed_sum = layer
        .codebook
        .embed_sum
        .val()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|source| Qwen3TtsInferenceError::TensorRead {
            message: format!("failed to read semantic codebook embeddings: {source}"),
        })?;

    let codebook_size = cluster_usage.len();
    if codebook_size == 0 || embed_sum.len() != codebook_size * hidden_size {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "semantic codebook tensor shapes are inconsistent".to_string(),
        });
    }

    let mut tokens = Vec::with_capacity(time_steps);
    let mut quantized_values = vec![0.0f32; hidden_size * time_steps];
    for time_index in 0..time_steps {
        let mut best_index = 0usize;
        let mut best_distance = f32::INFINITY;

        for code_index in 0..codebook_size {
            let usage = cluster_usage[code_index].max(1e-6);
            let mut distance = 0.0f32;
            for hidden_index in 0..hidden_size {
                let hidden_value = hidden_values[hidden_index * time_steps + time_index];
                let centroid = embed_sum[code_index * hidden_size + hidden_index] / usage;
                let diff = hidden_value - centroid;
                distance += diff * diff;
            }
            if distance < best_distance {
                best_distance = distance;
                best_index = code_index;
            }
        }

        tokens.push(best_index as i64);
        let usage = cluster_usage[best_index].max(1e-6);
        for hidden_index in 0..hidden_size {
            quantized_values[hidden_index * time_steps + time_index] =
                embed_sum[best_index * hidden_size + hidden_index] / usage;
        }
    }

    let quantized = Tensor::<B, 3>::from_data(
        TensorData::new(quantized_values, [1, hidden_size, time_steps]),
        &hidden.device(),
    );
    Ok((tokens, quantized))
}
