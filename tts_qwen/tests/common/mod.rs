use std::env;
use std::path::PathBuf;

use tts_qwen::{default_workspace_root, find_local_qwen_tts_model_dir};

pub fn resolve_model_dir() -> PathBuf {
    if let Ok(model_dir) = env::var("QWEN_TTS_MODEL_DIR") {
        let path = PathBuf::from(model_dir);
        assert!(
            path.join("config.json").is_file(),
            "QWEN_TTS_MODEL_DIR must point at a talker model directory containing config.json: {}",
            path.display()
        );
        return path;
    }

    find_local_qwen_tts_model_dir(default_workspace_root()).expect(
        "set QWEN_TTS_MODEL_DIR or place a local Qwen/* model directory under the workspace root",
    )
}

#[allow(dead_code)]
pub fn workspace_root() -> PathBuf {
    default_workspace_root()
}
