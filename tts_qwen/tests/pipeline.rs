mod common;

use burn::backend::Flex;
use burn::tensor::{Int, Tensor};
use tts_qwen::{
    CodePredictorGenerateInput, CustomVoiceBatch, CustomVoiceRequest, KeyValueCache,
    Qwen3TtsTextTokenizer, SamplingConfig, StoppingRules, TalkerGenerateInput,
    build_custom_voice_prefill_batch, decode_codec_tokens, generate_code_predictor_groups,
    generate_talker_tokens, load_custom_voice_generation_config, load_qwen3_tts_audio_codec,
    load_qwen3_tts_talker_for_inference, save_wav,
};

type Backend = Flex;

#[test]
fn pipeline_configuration_smoke() {
    let model_dir = common::resolve_model_dir();
    let config = load_custom_voice_generation_config(&model_dir).unwrap();

    assert_eq!(config.codec_eos_token_id, 2150);
    assert!(config.suppress_token_ids.contains(&2148));
    assert!(config.suppress_token_ids.contains(&2149));
    assert!(!config.suppress_token_ids.contains(&2150));
}

#[test]
#[ignore = "loads real Qwen weights and writes target/tmp/e2e/0000.wav"]
fn pipeline_generates_valid_wav_with_real_model() {
    let model_dir = common::resolve_model_dir();
    let output_dir = common::workspace_root().join("target/tmp/e2e");
    std::fs::create_dir_all(&output_dir).expect("e2e output dir should exist");
    let wav_path = output_dir.join("0000.wav");

    let device = Default::default();
    let talker = load_qwen3_tts_talker_for_inference::<Backend>(&model_dir, &device)
        .expect("talker should load");
    let audio_codec =
        load_qwen3_tts_audio_codec::<Backend>(&model_dir, &device).expect("codec should load");
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();
    let generation_config = load_custom_voice_generation_config(&model_dir).unwrap();
    let request = CustomVoiceRequest {
        text: "你好，欢迎使用语音合成。".to_string(),
        language: Some("Chinese".to_string()),
        speaker: Some("Vivian".to_string()),
    };
    let frontend = build_custom_voice_prefill_batch(
        &tokenizer,
        &talker.config.talker_config,
        &talker,
        &CustomVoiceBatch::single(request),
        &device,
    )
    .expect("frontend should build");

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
                max_new_tokens: 64,
                eos_token_id: Some(generation_config.codec_eos_token_id),
            },
            suppress_token_ids: generation_config.suppress_token_ids,
            collect_step_diagnostics: false,
        },
        &mut talker_cache,
    )
    .expect("talker should generate");

    let generated_ids = generated
        .generated_token_ids
        .clone()
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap();
    let time_steps = generated_audio_steps(&generated_ids, generation_config.codec_eos_token_id);
    assert!(
        time_steps > 0,
        "talker should emit at least one audio token"
    );

    let mut codec_steps = Vec::with_capacity(time_steps);
    for step in 0..time_steps {
        let base_token = generated
            .generated_token_ids
            .clone()
            .slice([0..1, step..step + 1]);
        let hidden = generated.step_hidden_states[step].clone();
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
        .expect("code predictor should generate");
        codec_steps.push(groups.codec_ids.reshape([1, cfg.num_code_groups, 1]));
    }

    let codec_tokens: Tensor<Backend, 3, Int> = Tensor::cat(codec_steps, 2);
    let waveform = decode_codec_tokens::<Backend>(
        &audio_codec,
        codec_tokens,
        &audio_codec.config.decoder_config,
    )
    .expect("codec should decode");
    save_wav(
        &waveform,
        &wav_path,
        audio_codec.config.output_sample_rate as u32,
    )
    .expect("wav should save");

    let wav = std::fs::read(&wav_path).expect("wav should be readable");
    assert_valid_nonempty_wav(&wav, audio_codec.config.output_sample_rate as u32);
}

fn generated_audio_steps(token_ids: &[i32], eos_token_id: usize) -> usize {
    token_ids
        .iter()
        .position(|id| *id as usize == eos_token_id)
        .unwrap_or(token_ids.len())
}

fn assert_valid_nonempty_wav(bytes: &[u8], expected_sample_rate: u32) {
    assert!(bytes.len() > 44, "wav must include header and PCM data");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "PCM format");
    assert_eq!(
        u16::from_le_bytes([bytes[22], bytes[23]]),
        1,
        "mono channel"
    );
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        expected_sample_rate
    );
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "16-bit PCM");
    assert_eq!(&bytes[36..40], b"data");
    let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    assert_eq!(bytes.len(), 44 + data_size);
    assert!(data_size > 0);
    assert!(
        bytes[44..].chunks_exact(2).any(|sample| sample != [0, 0]),
        "audio should not be all zero"
    );
}
