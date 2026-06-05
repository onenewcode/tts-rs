use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tts_qwen3_tts::{
    CustomVoiceRequest, FloatDType, LanguageSelection, Qwen3TtsEngine, Qwen3TtsEngineConfig,
    Qwen3TtsPackageSource, Qwen3TtsProfilingConfig, Qwen3TtsRunOptions, QwenRequest,
    SamplingOverride,
};

const MODEL_DIR: &str = "Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice";
const TEXT: &str = "你好，欢迎使用 tts-rs。";
const LANGUAGE: &str = "Chinese";
const SPEAKER: &str = "Vivian";
const MAX_NEW_TOKENS: usize = 32;
const WARMUP_SYNTHESIS_RUNS: usize = 0;
const MEASURED_SYNTHESIS_RUNS: usize = 1;

pub fn run_custom_voice_bf16() -> Result<()> {
    println!("qwen3 custom-voice bf16 benchmark");
    println!("model_dir={}", workspace_path(MODEL_DIR).display());
    println!("text={TEXT}");
    println!("language={LANGUAGE} speaker={SPEAKER} max_new_tokens={MAX_NEW_TOKENS}");
    println!(
        "warmup_synthesis_runs={WARMUP_SYNTHESIS_RUNS} measured_synthesis_runs={MEASURED_SYNTHESIS_RUNS}"
    );
    flush_stdout()?;

    let load_started = Instant::now();
    let engine = load_engine()?;
    let load_elapsed = load_started.elapsed();
    println!("load_ms={:.3}", millis(load_elapsed));
    flush_stdout()?;

    let request = custom_voice_request();
    let options = run_options();

    for index in (0usize..).take(WARMUP_SYNTHESIS_RUNS) {
        let sample = synthesize_once(&engine, &request, &options)
            .with_context(|| format!("warmup synthesis {} failed", index + 1))?;
        println!(
            "warmup={} synthesis_ms={:.3} audio_s={:.3} rtf={:.3}",
            index + 1,
            millis(sample.elapsed),
            sample.audio_seconds,
            sample.rtf
        );
        flush_stdout()?;
    }

    let mut measured = Vec::with_capacity(MEASURED_SYNTHESIS_RUNS);
    for index in 0..MEASURED_SYNTHESIS_RUNS {
        let sample = synthesize_once(&engine, &request, &options)
            .with_context(|| format!("measured synthesis {} failed", index + 1))?;
        println!(
            "measure={} synthesis_ms={:.3} audio_s={:.3} rtf={:.3}",
            index + 1,
            millis(sample.elapsed),
            sample.audio_seconds,
            sample.rtf
        );
        flush_stdout()?;
        measured.push(sample);
    }

    let summary = summarize(&measured)?;
    println!(
        "summary load_ms={:.3} synthesis_ms={:.3} audio_s={:.3} rtf={:.3}",
        millis(load_elapsed),
        millis(summary.elapsed),
        summary.audio_seconds,
        summary.rtf
    );
    Ok(())
}

struct SynthesisSample {
    elapsed: Duration,
    audio_seconds: f64,
    rtf: f64,
}

fn load_engine() -> Result<Qwen3TtsEngine> {
    Qwen3TtsEngine::load(Qwen3TtsEngineConfig {
        package: Qwen3TtsPackageSource::ModelDir(workspace_path(MODEL_DIR)),
        profiling: Qwen3TtsProfilingConfig::default(),
        talker_dtype: Some(FloatDType::BF16),
        codec_dtype: Some(FloatDType::BF16),
    })
    .context("failed to load Qwen3-TTS custom-voice model")
}

fn synthesize_once(
    engine: &Qwen3TtsEngine,
    request: &QwenRequest,
    options: &Qwen3TtsRunOptions,
) -> Result<SynthesisSample> {
    let started = Instant::now();
    let audio = engine
        .synthesize(request.clone(), options.clone())
        .context("synthesis failed")?;
    let elapsed = started.elapsed();
    if audio.sample_rate == 0 || audio.channels == 0 || audio.pcm_i16.is_empty() {
        bail!("generated invalid audio metadata");
    }
    let audio_seconds =
        audio.pcm_i16.len() as f64 / f64::from(audio.sample_rate) / f64::from(audio.channels);
    Ok(SynthesisSample {
        elapsed,
        audio_seconds,
        rtf: elapsed.as_secs_f64() / audio_seconds,
    })
}

fn summarize(samples: &[SynthesisSample]) -> Result<SynthesisSample> {
    if samples.is_empty() {
        bail!("at least one measured synthesis run is required");
    }
    let elapsed = samples
        .iter()
        .map(|sample| sample.elapsed)
        .sum::<Duration>()
        / u32::try_from(samples.len()).expect("sample count should fit in u32");
    let audio_seconds = samples
        .iter()
        .map(|sample| sample.audio_seconds)
        .sum::<f64>()
        / samples.len() as f64;
    Ok(SynthesisSample {
        elapsed,
        audio_seconds,
        rtf: elapsed.as_secs_f64() / audio_seconds,
    })
}

fn custom_voice_request() -> QwenRequest {
    QwenRequest::CustomVoice(CustomVoiceRequest {
        text: TEXT.to_string(),
        language: LanguageSelection::Named(LANGUAGE.to_string()),
        speaker: Some(SPEAKER.to_string()),
        instruct: None,
    })
}

fn run_options() -> Qwen3TtsRunOptions {
    Qwen3TtsRunOptions {
        max_new_tokens: Some(MAX_NEW_TOKENS),
        talker_sampling: Some(SamplingOverride::GreedyFromModelDefaults),
        code_predictor_sampling: Some(SamplingOverride::GreedyFromModelDefaults),
    }
}

fn workspace_path(relative: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn flush_stdout() -> io::Result<()> {
    io::stdout().flush()
}
