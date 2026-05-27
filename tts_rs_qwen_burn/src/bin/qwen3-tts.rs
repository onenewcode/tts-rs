use std::path::PathBuf;

use burn::backend::Flex;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use serde::Serialize;
use tts_rs_qwen_burn::{
    CodePredictorGenerateInput, CustomVoiceBatch, CustomVoiceRequest, KeyValueCache,
    Qwen3TtsTextTokenizer, SamplingConfig, StoppingRules, TalkerGenerateInput,
    build_custom_voice_prefill_batch, decode_codec_tokens, generate_code_predictor_groups,
    generate_talker_tokens, load_custom_voice_generation_config, load_qwen3_tts_audio_codec,
    load_qwen3_tts_talker_for_inference, save_wav,
};

type Backend = Flex;

#[derive(Debug, Serialize)]
struct OutputManifest {
    sample_rate: u32,
    files: Vec<String>,
    text: String,
    language: Option<String>,
    speaker: Option<String>,
    max_new_tokens: usize,
    talker_token_count: usize,
    talker_token_ids: Vec<i32>,
    codec_time_steps: usize,
    codec_first_frame: Vec<i32>,
    codec_preview_frames: Vec<Vec<i32>>,
    waveform: WaveformStats,
    audio_status: AudioStatus,
}

#[derive(Debug, Serialize)]
struct WaveformStats {
    shape: [usize; 3],
    sample_count: usize,
    duration_sec: f32,
    min: f32,
    max: f32,
    peak: f32,
    rms: f32,
    mean: f32,
    clip_fraction: f32,
}

#[derive(Debug, Serialize)]
struct AudioStatus {
    valid_container_written: bool,
    non_empty: bool,
    has_signal: bool,
    likely_clipped: bool,
    verdict: String,
}

#[derive(Debug, Parser)]
#[command(name = "qwen3-tts")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    text: String,
    #[arg(long)]
    language: Option<String>,
    #[arg(long)]
    speaker: Option<String>,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&args.model_dir, &device)
        .map_err(|e| format!("failed to load talker: {e}"))?;
    let audio_codec = load_qwen3_tts_audio_codec::<Backend>(&args.model_dir, &device)
        .map_err(|e| format!("failed to load audio codec: {e}"))?;
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&args.model_dir)
        .map_err(|e| format!("failed to load text tokenizer: {e}"))?;
    let generation_config = load_custom_voice_generation_config(&args.model_dir)
        .map_err(|e| format!("failed to load generation config: {e}"))?;

    let request = CustomVoiceRequest {
        text: args.text.clone(),
        language: args.language.clone(),
        speaker: args.speaker.clone(),
    };
    let frontend = build_custom_voice_prefill_batch(
        &tokenizer,
        &talker.config.talker_config,
        &talker,
        &CustomVoiceBatch::single(request),
        &device,
    )
    .map_err(|e| format!("failed to build frontend prefill: {e}"))?;

    let cfg = &talker.config.talker_config;
    let mut talker_cache = (0..cfg.num_hidden_layers)
        .map(|_| KeyValueCache::new(1, cfg.num_key_value_heads, 4096, cfg.head_dim))
        .collect::<Vec<_>>();
    let generated = generate_talker_tokens(
        cfg,
        &talker,
        TalkerGenerateInput {
            prefill_inputs_embeds: frontend.inputs_embeds,
            prefill_position_ids: frontend.position_ids,
            prefill_attention_mask: Some(frontend.attention_mask),
            trailing_text_hidden: Some(frontend.trailing_text_hidden),
            tts_pad_embed: Some(frontend.tts_pad_embed),
            sampling: SamplingConfig::greedy(),
            stopping: StoppingRules {
                max_new_tokens: args.max_new_tokens,
                eos_token_id: Some(generation_config.codec_eos_token_id),
            },
            suppress_token_ids: generation_config.suppress_token_ids.clone(),
            collect_step_diagnostics: false,
        },
        &mut talker_cache,
    )
    .map_err(|e| format!("talker generation failed: {e}"))?;

    let generated_token_ids = generated
        .generated_token_ids
        .clone()
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .map_err(|e| format!("failed to read generated token ids: {e}"))?;
    let time_steps = generated_token_ids
        .iter()
        .position(|id| *id as usize == generation_config.codec_eos_token_id)
        .unwrap_or(generated_token_ids.len());
    if time_steps == 0 {
        return Err("talker emitted EOS before any audio codec token".into());
    }
    let mut codec_steps = Vec::with_capacity(time_steps);
    let mut codec_frames = Vec::with_capacity(time_steps);
    for t in 0..time_steps {
        let base_token = generated
            .generated_token_ids
            .clone()
            .slice([0..1, t..t + 1]);
        let hidden = generated.step_hidden_states[t].clone();
        let mut predictor_cache = (0..cfg.code_predictor_config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    cfg.code_predictor_config.num_key_value_heads,
                    cfg.num_code_groups + 1,
                    cfg.code_predictor_config.head_dim,
                )
            })
            .collect::<Vec<_>>();
        let groups = generate_code_predictor_groups(
            cfg,
            &talker,
            CodePredictorGenerateInput {
                talker_hidden_state: hidden,
                base_codec_token_id: base_token,
                sampling: SamplingConfig::greedy(),
                collect_step_diagnostics: false,
            },
            &mut predictor_cache,
        )
        .map_err(|e| format!("code predictor generation failed at step {t}: {e}"))?;
        let frame = groups
            .codec_ids
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .map_err(|e| format!("failed to read codec frame {t}: {e}"))?;
        codec_frames.push(frame);
        codec_steps.push(groups.codec_ids.reshape([1, cfg.num_code_groups, 1]));
    }
    let codec_tokens: Tensor<Backend, 3, Int> = Tensor::cat(codec_steps, 2);
    let waveform = decode_codec_tokens::<Backend>(
        &audio_codec,
        codec_tokens,
        &audio_codec.config.decoder_config,
    )
    .map_err(|e| format!("audio codec decoding failed: {e}"))?;
    let waveform_stats = waveform_stats(&waveform, audio_codec.config.output_sample_rate as u32)?;

    let wav_path = args.output_dir.join("0000.wav");
    save_wav(
        &waveform,
        &wav_path,
        audio_codec.config.output_sample_rate as u32,
    )?;
    let audio_status = classify_audio(&waveform_stats);
    let manifest = OutputManifest {
        sample_rate: audio_codec.config.output_sample_rate as u32,
        files: vec!["0000.wav".to_string()],
        text: args.text,
        language: args.language,
        speaker: args.speaker,
        max_new_tokens: args.max_new_tokens,
        talker_token_count: time_steps,
        talker_token_ids: generated_token_ids[..time_steps].to_vec(),
        codec_time_steps: time_steps,
        codec_first_frame: codec_frames.first().cloned().unwrap_or_default(),
        codec_preview_frames: codec_frames.iter().take(4).cloned().collect(),
        waveform: waveform_stats,
        audio_status,
    };
    std::fs::write(
        args.output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn waveform_stats<B: burn::tensor::backend::Backend>(
    waveform: &Tensor<B, 3>,
    sample_rate: u32,
) -> Result<WaveformStats, Box<dyn std::error::Error>> {
    let shape = waveform.dims();
    let samples = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| format!("failed to read waveform samples: {e}"))?;
    let sample_count = samples.len();
    if sample_count == 0 {
        return Ok(WaveformStats {
            shape,
            sample_count,
            duration_sec: 0.0,
            min: 0.0,
            max: 0.0,
            peak: 0.0,
            rms: 0.0,
            mean: 0.0,
            clip_fraction: 0.0,
        });
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut clipped = 0usize;
    for sample in &samples {
        min = min.min(*sample);
        max = max.max(*sample);
        sum += *sample as f64;
        sum_sq += (*sample as f64) * (*sample as f64);
        if sample.abs() >= 0.999 {
            clipped += 1;
        }
    }
    let peak = min.abs().max(max.abs());
    Ok(WaveformStats {
        shape,
        sample_count,
        duration_sec: sample_count as f32 / sample_rate as f32,
        min,
        max,
        peak,
        rms: (sum_sq / sample_count as f64).sqrt() as f32,
        mean: (sum / sample_count as f64) as f32,
        clip_fraction: clipped as f32 / sample_count as f32,
    })
}

fn classify_audio(stats: &WaveformStats) -> AudioStatus {
    let non_empty = stats.sample_count > 0 && stats.duration_sec > 0.0;
    let has_signal = stats.rms > 1.0e-4 && stats.peak > 1.0e-3;
    let likely_clipped = stats.clip_fraction > 0.01 || stats.peak > 1.5;
    let verdict = if !non_empty {
        "invalid_empty"
    } else if !has_signal {
        "invalid_silent"
    } else if likely_clipped {
        "invalid_clipped"
    } else {
        "plausible"
    };
    AudioStatus {
        valid_container_written: true,
        non_empty,
        has_signal,
        likely_clipped,
        verdict: verdict.to_string(),
    }
}
