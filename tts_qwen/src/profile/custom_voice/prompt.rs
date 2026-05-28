use super::request::CustomVoiceRequest;

pub fn build_custom_voice_prompt(request: &CustomVoiceRequest) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        request.text
    )
}
