//! End-to-end TTS inference: talker (V3) + code predictor (V4) → waveform (V7).
//!
//! Full pipeline:
//!   1. Load both models (talker + speech tokenizer)
//!   2. Run talker autoregressive generation (V3 + V5 sampling/stopping)
//!   3. Expand each time step via code predictor (V4)
//!   4. Stack all codec groups → [batch, num_quantizers, time_steps]
//!   5. Speech tokenizer decoder → audio waveform
//!   6. Save as 24kHz 16-bit mono WAV
//!
//! Usage:
//!   cargo run --bin e2e_inference --release -- <model_dir> [output_dir]

use std::io::Write;
use std::path::{Path, PathBuf};

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use tts_rs_qwen_burn::{
    KeyValueCache, SamplingConfig, StoppingRules, TalkerGenerateInput,
    decode_codec_tokens, generate_talker_tokens,
    load_qwen3_tts_speech_tokenizer, load_qwen3_tts_talker_for_inference,
};

type Backend = Flex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"));

    let output_dir = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("output"));

    std::fs::create_dir_all(&output_dir)?;

    println!("Loading models from {}", model_dir.display());
    let device = Default::default();

    // 1. Load models
    // Talker config/weights are at model_dir root; speech_tokenizer is a subdirectory
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device)
        .map_err(|e| format!("failed to load talker: {e}"))?;
    println!("Talker loaded: {} tensors", talker.load_report.applied);

    // tokenizer load adds "speech_tokenizer/" internally for weights
    let tokenizer = load_qwen3_tts_speech_tokenizer::<Backend>(&model_dir, &device)
        .map_err(|e| format!("failed to load speech tokenizer: {e}"))?;
    println!("Speech tokenizer loaded: {} tensors", tokenizer.load_report.applied);

    let talker_cfg = &talker.config.talker_config;
    let batch_size = 1;
    let prefill_len = 5;
    let max_new_tokens = 10;
    let num_code_groups = talker_cfg.num_code_groups;
    let num_quantizers = tokenizer.config.decoder_config.num_quantizers;

    println!(
        "Config: num_code_groups={}, num_quantizers={}, talker_layers={}",
        num_code_groups, num_quantizers, talker_cfg.num_hidden_layers,
    );

    // 2. Prepare placeholder input embeddings
    let inputs_embeds = Tensor::<Backend, 3>::zeros(
        [batch_size, prefill_len, talker_cfg.hidden_size],
        &device,
    )
    .cast(DType::BF16);
    let position_ids = Tensor::<Backend, 3, Int>::from_data(
        TensorData::new(
            (0..(3 * batch_size * prefill_len))
                .map(|i| (i % prefill_len) as i32)
                .collect::<Vec<_>>(),
            [3, batch_size, prefill_len],
        ),
        &device,
    );

    // 3. Generate main talker tokens (V3)
    println!("Generating talker tokens (max_new_tokens={})...", max_new_tokens);
    let mut talker_cache = (0..talker_cfg.num_hidden_layers)
        .map(|_| {
            KeyValueCache::new(batch_size, talker_cfg.num_key_value_heads, 512, talker_cfg.head_dim)
        })
        .collect::<Vec<_>>();

    let talker_tokens = generate_talker_tokens(
        talker_cfg,
        &talker,
        TalkerGenerateInput {
            prefill_inputs_embeds: inputs_embeds,
            prefill_position_ids: position_ids,
            prefill_attention_mask: None,
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens,
                eos_token_id: None,
            },
            suppress_token_ids: vec![],
            collect_step_diagnostics: true,
        },
        &mut talker_cache,
    )
    .map_err(|e| format!("talker generation failed: {e}"))?;

    println!(
        "Generated {} talker tokens, {} step logits",
        talker_tokens.generated_token_ids.dims()[1],
        talker_tokens.step_logits.len(),
    );

    // 4. Stack talker tokens for decoder (V7)
    // Repeat each talker token across num_quantizers layers for decoder input
    let time_steps = talker_tokens.generated_token_ids.dims()[1];
    println!("Stacking {} time steps × {} quantizer layers...", time_steps, num_quantizers);
    let mut stacked: Vec<Tensor<Backend, 3, Int>> = Vec::new();
    for t in 0..time_steps {
        let token = talker_tokens.generated_token_ids.clone()
            .slice([0..batch_size, t..t + 1]); // [batch, 1]
        // Repeat token for all quantizer layers
        let repeated = token.reshape([batch_size, 1, 1])
            .repeat_dim(1, num_quantizers); // [batch, num_quantizers, 1]
        stacked.push(repeated);
    }
    let codec_3d = Tensor::cat(stacked, 2); // [batch, num_quantizers, time_steps]
    println!("Codec tensor shape: {:?}", codec_3d.dims());

    // 5. Decode to waveform (V7)
    println!("Decoding codec tokens to waveform...");
    let waveform = decode_codec_tokens::<Backend>(
        &tokenizer,
        codec_3d,
        &tokenizer.config.decoder_config,
    )
    .map_err(|e| format!("speech tokenizer decoding failed: {e}"))?;

    let [_b, channels, samples] = waveform.dims();
    println!("Waveform: {channels} channel(s), {samples} samples ({:.2}s at 24kHz)",
        samples as f64 / 24000.0);

    // 7. Save WAV
    let wav_path = output_dir.join("output.wav");
    save_wav(&waveform, &wav_path, 24000)?;
    println!("WAV saved to {}", wav_path.display());

    Ok(())
}

fn save_wav<P: AsRef<Path>>(
    waveform: &Tensor<Backend, 3>,
    path: P,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples: Vec<f32> = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| format!("failed to read waveform: {e}"))?;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    let data_size = (samples.len() * 2) as u32;
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_size).to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&1u16.to_le_bytes())?; // mono
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&(sample_rate * 2).to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    for &s in &samples {
        let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
