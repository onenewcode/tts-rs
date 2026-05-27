use std::path::PathBuf;

use burn::backend::Flex;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use tts_qwen::{
    CodePredictorGenerateInput, CustomVoiceBatch, CustomVoiceRequest, KeyValueCache,
    Qwen3TtsTextTokenizer, SamplingConfig, StoppingRules, TalkerGenerateInput,
    build_custom_voice_prefill_batch, decode_codec_tokens, generate_code_predictor_groups,
    generate_talker_tokens, load_custom_voice_generation_config, load_qwen3_tts_audio_codec,
    load_qwen3_tts_talker_for_inference, save_wav,
};

type Backend = Flex;

#[derive(Debug, Parser)]
#[command(name = "tts_cli")]
pub struct Args {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub text: String,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long)]
    pub speaker: Option<String>,
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
}

pub fn run_from_args() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
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
    let time_steps =
        generated_audio_steps(&generated_token_ids, generation_config.codec_eos_token_id);
    if time_steps == 0 {
        return Err("talker emitted EOS before any audio codec token".into());
    }

    let mut codec_steps = Vec::with_capacity(time_steps);
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
        codec_steps.push(groups.codec_ids.reshape([1, cfg.num_code_groups, 1]));
    }

    let codec_tokens: Tensor<Backend, 3, Int> = Tensor::cat(codec_steps, 2);
    let waveform = decode_codec_tokens::<Backend>(
        &audio_codec,
        codec_tokens,
        &audio_codec.config.decoder_config,
    )
    .map_err(|e| format!("audio codec decoding failed: {e}"))?;

    let wav_path = args.output_dir.join("0000.wav");
    save_wav(
        &waveform,
        &wav_path,
        audio_codec.config.output_sample_rate as u32,
    )?;
    Ok(())
}

fn generated_audio_steps(token_ids: &[i32], eos_token_id: usize) -> usize {
    token_ids
        .iter()
        .position(|id| *id as usize == eos_token_id)
        .unwrap_or(token_ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parse_required_fields_with_defaults() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--model-dir",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "--text",
            "hello",
        ])
        .expect("minimal args should parse");

        assert_eq!(
            args.model_dir,
            PathBuf::from("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        );
        assert_eq!(args.text, "hello");
        assert_eq!(args.language, None);
        assert_eq!(args.speaker, None);
        assert_eq!(args.output_dir, PathBuf::from("output"));
        assert_eq!(args.max_new_tokens, 256);
    }

    #[test]
    fn args_parse_optional_generation_fields() {
        let args = Args::try_parse_from([
            "tts_cli",
            "--model-dir",
            "model",
            "--text",
            "你好",
            "--language",
            "Chinese",
            "--speaker",
            "Vivian",
            "--output-dir",
            ".",
            "--max-new-tokens",
            "64",
        ])
        .expect("full args should parse");

        assert_eq!(args.language.as_deref(), Some("Chinese"));
        assert_eq!(args.speaker.as_deref(), Some("Vivian"));
        assert_eq!(args.output_dir, PathBuf::from("."));
        assert_eq!(args.max_new_tokens, 64);
    }

    #[test]
    fn generated_audio_steps_stop_before_eos() {
        assert_eq!(generated_audio_steps(&[10, 11, 2150, 12], 2150), 2);
    }

    #[test]
    fn generated_audio_steps_keep_all_tokens_when_eos_absent() {
        assert_eq!(generated_audio_steps(&[10, 11, 12], 2150), 3);
    }
}
