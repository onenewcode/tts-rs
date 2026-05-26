mod common;

use std::process::Command;

use serde::Deserialize;
use tts_rs_qwen_burn::{CustomVoiceRequest, Qwen3TtsTextTokenizer, build_custom_voice_prompt};

#[derive(Debug, Deserialize)]
struct TokenizerReference {
    prompt: String,
    token_ids: Vec<i64>,
}

#[test]
fn tokenizer_matches_python_oracle() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_tokenizer.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_tokenizer.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python tokenizer oracle");
    assert!(status.success(), "Python tokenizer oracle failed");

    let reference: TokenizerReference =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();
    let tokenizer = Qwen3TtsTextTokenizer::from_model_dir(&model_dir).unwrap();
    let request = CustomVoiceRequest::new("其实我真的有发现，我是一个特别善于观察别人情绪的人。");
    assert_eq!(build_custom_voice_prompt(&request), reference.prompt);
    assert_eq!(
        tokenizer.encode(&reference.prompt).unwrap(),
        reference.token_ids
    );
}
