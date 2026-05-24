use std::fs;
use std::path::{Path, PathBuf};

use crate::{Qwen3TtsLoadError, Qwen3TtsVerifyError};

pub fn default_workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn find_local_qwen_tts_model_dir(
    workspace_root: impl AsRef<Path>,
) -> Result<PathBuf, Qwen3TtsLoadError> {
    let root = workspace_root.as_ref().join("Qwen");
    let entries = fs::read_dir(&root).map_err(|source| Qwen3TtsLoadError::Io {
        path: root.clone(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| Qwen3TtsLoadError::Io {
            path: root.clone(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir()
            && path.join("config.json").is_file()
            && path.join("model.safetensors").is_file()
        {
            return Ok(path);
        }
    }

    Err(Qwen3TtsLoadError::ModelDirNotFound { root })
}

pub fn ensure_parent_dir(path: impl AsRef<Path>) -> Result<(), Qwen3TtsVerifyError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| Qwen3TtsVerifyError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}
