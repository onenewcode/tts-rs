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

#[cfg(test)]
mod tests {
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("valid system time")
                .as_nanos();
            let path =
                env::temp_dir().join(format!("tts-rs-{name}-{}-{unique}", std::process::id()));
            fs::create_dir_all(&path).expect("temp dir should be created");
            Self { path }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn default_workspace_root_is_crate_parent() {
        assert_eq!(
            default_workspace_root(),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
        );
    }

    #[test]
    fn ensure_parent_dir_creates_nested_directories() {
        let temp_dir = TempDirGuard::new("ensure-parent-dir");
        let target = temp_dir.path.join("a/b/c/report.json");

        ensure_parent_dir(&target).expect("parent directories should be created");

        assert!(temp_dir.path.join("a/b/c").is_dir());
    }

    #[test]
    fn find_local_qwen_tts_model_dir_selects_directory_with_required_files() {
        let temp_dir = TempDirGuard::new("find-model-dir");
        let qwen_root = temp_dir.path.join("Qwen");
        let invalid_dir = qwen_root.join("invalid");
        let valid_dir = qwen_root.join("valid");

        fs::create_dir_all(&invalid_dir).expect("invalid dir should exist");
        fs::create_dir_all(&valid_dir).expect("valid dir should exist");
        fs::write(invalid_dir.join("config.json"), "{}").expect("invalid config should exist");
        fs::write(valid_dir.join("config.json"), "{}").expect("valid config should exist");
        fs::write(valid_dir.join("model.safetensors"), b"stub")
            .expect("valid weights should exist");

        let model_dir =
            find_local_qwen_tts_model_dir(&temp_dir.path).expect("model dir should be found");

        assert_eq!(model_dir, valid_dir);
    }

    #[test]
    fn find_local_qwen_tts_model_dir_returns_not_found_when_qwen_dir_has_no_complete_model() {
        let temp_dir = TempDirGuard::new("find-model-dir-missing");
        let qwen_root = temp_dir.path.join("Qwen");
        let partial_dir = qwen_root.join("partial");

        fs::create_dir_all(&partial_dir).expect("partial dir should exist");
        fs::write(partial_dir.join("config.json"), "{}").expect("config should exist");

        let error = find_local_qwen_tts_model_dir(&temp_dir.path).expect_err("should fail");

        assert!(matches!(error, Qwen3TtsLoadError::ModelDirNotFound { .. }));
    }

    #[test]
    fn find_local_qwen_tts_model_dir_returns_io_error_when_qwen_root_is_missing() {
        let temp_dir = TempDirGuard::new("find-model-dir-io");

        let error = find_local_qwen_tts_model_dir(&temp_dir.path).expect_err("should fail");

        assert!(matches!(error, Qwen3TtsLoadError::Io { .. }));
    }
}
