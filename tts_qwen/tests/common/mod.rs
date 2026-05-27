use std::env;
use std::path::PathBuf;

pub fn resolve_model_dir() -> PathBuf {
    let path =
        PathBuf::from(env::var("QWEN_TTS_MODEL_DIR").expect("QWEN_TTS_MODEL_DIR must be set"));
    assert!(
        path.join("config.json").is_file(),
        "QWEN_TTS_MODEL_DIR must point at a talker model directory containing config.json: {}",
        path.display()
    );
    path
}

#[allow(dead_code)]
pub fn unique_output_dir(name: &str) -> PathBuf {
    let unique = format!("{}-{}", std::process::id(), chrono_like_timestamp());
    env::temp_dir().join(format!("tts-rs-{name}-{unique}"))
}

fn chrono_like_timestamp() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos()
}
