use std::path::PathBuf;

use burn::backend::Flex;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use tts_rs_qwen_burn::{
    build_custom_voice_prefill_batch, decode_codec_tokens, generate_code_predictor_groups,
    generate_talker_tokens, load_custom_voice_generation_config, load_qwen3_tts_audio_codec,
    load_qwen3_tts_talker_for_inference, save_wav, CodePredictorGenerateInput, CustomVoiceBatch,
    CustomVoiceRequest, KeyValueCache, Qwen3TtsTextTokenizer, SamplingConfig, StoppingRules,
    TalkerGenerateInput,
};

type Backend = Flex;

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
        text: args.text,
        language: args.language,
        speaker: args.speaker,
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
    save_wav(&waveform, &wav_path, audio_codec.config.output_sample_rate as u32)?;
    std::fs::write(
        args.output_dir.join("manifest.json"),
        format!(
            "{{\n  \"sample_rate\": {},\n  \"files\": [\"0000.wav\"]\n}}\n",
            audio_codec.config.output_sample_rate
        ),
    )?;
    Ok(())
}
