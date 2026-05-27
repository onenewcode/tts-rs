mod common;

use tts_qwen::{
    CustomVoiceRequest, build_custom_voice_prompt, load_custom_voice_generation_config,
};

#[test]
fn custom_voice_prompt_matches_qwen_tts_template() {
    let request = CustomVoiceRequest::new("hello");
    assert_eq!(
        build_custom_voice_prompt(&request),
        "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn generation_config_uses_codec_eos_and_suppresses_reserved_range() {
    let config = load_custom_voice_generation_config(&common::resolve_model_dir()).unwrap();
    assert_eq!(config.codec_eos_token_id, 2150);
    assert!(config.suppress_token_ids.contains(&2148));
    assert!(config.suppress_token_ids.contains(&2149));
    assert!(!config.suppress_token_ids.contains(&2150));
}
