use super::request::BaseRequest;

pub(crate) fn build_base_prompt(request: &BaseRequest) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        request.text
    )
}
