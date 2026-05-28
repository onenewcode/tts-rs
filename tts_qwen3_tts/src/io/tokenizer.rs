use std::path::Path;

use serde::Deserialize;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: serde_json::Map<String, serde_json::Value>,
}

pub fn load_qwen3_tts_tokenizer(model_dir: impl AsRef<Path>) -> tokenizers::Result<Tokenizer> {
    let model_dir = model_dir.as_ref();
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let bpe = BPE::from_file(
        vocab.to_string_lossy().as_ref(),
        merges.to_string_lossy().as_ref(),
    )
    .unk_token("<|endoftext|>".to_string())
    .build()?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_special_tokens(read_special_tokens(model_dir)?)?;
    Ok(tokenizer)
}

fn read_special_tokens(model_dir: &Path) -> tokenizers::Result<Vec<AddedToken>> {
    let path = model_dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&path)?;
    let config: TokenizerConfig = serde_json::from_str(&text)?;
    let mut tokens = Vec::new();
    for value in config.added_tokens_decoder.values() {
        let Some(content) = value.get("content").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let special = value
            .get("special")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let normalized = value
            .get("normalized")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(!special);
        let lstrip = value
            .get("lstrip")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let rstrip = value
            .get("rstrip")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let single_word = value
            .get("single_word")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        tokens.push(
            AddedToken::from(content.to_string(), special)
                .normalized(normalized)
                .lstrip(lstrip)
                .rstrip(rstrip)
                .single_word(single_word),
        );
    }
    Ok(tokens)
}
