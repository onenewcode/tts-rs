mod common;

use tts_qwen::{CustomVoiceRequest, Qwen3TtsTextTokenizer, build_custom_voice_prompt};

const SAMPLE_TEXT: &str = "其实我真的有发现，我是一个特别善于观察别人情绪的人。";
const SAMPLE_PROMPT: &str = "<|im_start|>assistant\n其实我真的有发现，我是一个特别善于观察别人情绪的人。<|im_end|>\n<|im_start|>assistant\n";
const SAMPLE_TOKEN_IDS: &[i64] = &[
    151644, 77091, 198, 101045, 110146, 18830, 99879, 3837, 35946, 101909, 100654, 106614, 104144,
    101106, 104405, 100623, 1773, 151645, 198, 151644, 77091, 198,
];

#[test]
fn custom_voice_prompt_has_stable_chat_template() {
    let request = CustomVoiceRequest::new(SAMPLE_TEXT);
    assert_eq!(build_custom_voice_prompt(&request), SAMPLE_PROMPT);
}

#[test]
fn tokenizer_encodes_custom_voice_prompt_with_rust_golden_ids() {
    let model_dir = common::resolve_model_dir();
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();

    assert_eq!(tokenizer.encode(SAMPLE_PROMPT).unwrap(), SAMPLE_TOKEN_IDS);
}

#[test]
fn tokenizer_registers_required_special_tokens() {
    let model_dir = common::resolve_model_dir();
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();

    assert_eq!(tokenizer.encode("<|im_start|>").unwrap(), vec![151644]);
    assert_eq!(tokenizer.encode("<|im_end|>").unwrap(), vec![151645]);
    assert_eq!(tokenizer.encode("<tts_text_bos>").unwrap(), vec![151672]);
    assert_eq!(tokenizer.encode("<tts_text_eod>").unwrap(), vec![151673]);
    assert_eq!(tokenizer.encode("<tts_pad>").unwrap(), vec![151671]);
}
