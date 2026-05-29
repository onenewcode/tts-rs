use std::path::{Path, PathBuf};

use crate::Qwen3TtsLoadError;

use super::manifest::{Qwen3TtsGenerationConfigManifest, Qwen3TtsPackageManifest};

const PACKAGE_MANIFEST_FORMAT: &str = "qwen3_tts_package/v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen3TtsPackageSource {
    ManifestPath(PathBuf),
    ModelDir(PathBuf),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsPackage {
    pub package_root: PathBuf,
    pub name: String,
    pub tokenizer_path: PathBuf,
    pub talker_config_path: PathBuf,
    pub talker_weights_path: PathBuf,
    pub generation_config: Qwen3TtsGenerationConfigSource,
    pub codec_config_path: PathBuf,
    pub codec_weights_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qwen3TtsGenerationConfigSource {
    Path(PathBuf),
    Inline(Qwen3TtsGenerationConfigManifest),
}

impl Qwen3TtsPackage {
    pub fn load(source: &Qwen3TtsPackageSource) -> Result<Self, Qwen3TtsLoadError> {
        match source {
            Qwen3TtsPackageSource::ManifestPath(path) => Self::load_manifest(path),
            Qwen3TtsPackageSource::ModelDir(path) => Self::load_model_dir(path),
        }
    }

    fn load_manifest(manifest_path: &Path) -> Result<Self, Qwen3TtsLoadError> {
        let manifest_path = manifest_path.to_path_buf();
        let package_root = manifest_path
            .parent()
            .ok_or_else(|| Qwen3TtsLoadError::InvalidManifest {
                message: format!(
                    "package manifest path has no parent: {}",
                    manifest_path.display()
                ),
            })?
            .to_path_buf();

        let raw =
            std::fs::read_to_string(&manifest_path).map_err(|source| Qwen3TtsLoadError::Io {
                path: manifest_path.clone(),
                source,
            })?;
        let manifest: Qwen3TtsPackageManifest =
            serde_yaml::from_str(&raw).map_err(|source| Qwen3TtsLoadError::ManifestParse {
                path: manifest_path.clone(),
                source,
            })?;

        if manifest.format != PACKAGE_MANIFEST_FORMAT {
            return Err(Qwen3TtsLoadError::InvalidManifest {
                message: format!(
                    "unsupported package format `{}`; expected {PACKAGE_MANIFEST_FORMAT}",
                    manifest.format
                ),
            });
        }

        Ok(Self {
            package_root: package_root.clone(),
            name: manifest.name,
            tokenizer_path: resolve_path(&package_root, &manifest.artifacts.tokenizer),
            talker_config_path: resolve_path(&package_root, &manifest.artifacts.talker_config),
            talker_weights_path: resolve_path(&package_root, &manifest.artifacts.talker_weights),
            generation_config: Qwen3TtsGenerationConfigSource::Inline(manifest.generation_config),
            codec_config_path: resolve_path(&package_root, &manifest.artifacts.codec_config),
            codec_weights_path: resolve_path(&package_root, &manifest.artifacts.codec_weights),
        })
    }

    fn load_model_dir(model_dir: &Path) -> Result<Self, Qwen3TtsLoadError> {
        let model_dir = model_dir.to_path_buf();
        let tokenizer_path =
            match first_existing_file(&model_dir, &["tokenizer.json", "vocab.json"]) {
                Some(path) => path,
                None => {
                    return Err(Qwen3TtsLoadError::InvalidModelDir {
                        message: format!(
                            "{} is missing tokenizer.json or vocab.json",
                            model_dir.display()
                        ),
                    });
                }
            };

        let talker_config_path = require_file(&model_dir, "config.json")?;
        let talker_weights_path = require_file(&model_dir, "model.safetensors")?;
        let generation_config_path = require_file(&model_dir, "generation_config.json")?;
        let codec_config_path = require_file(&model_dir, "speech_tokenizer/config.json")?;
        let codec_weights_path = require_file(&model_dir, "speech_tokenizer/model.safetensors")?;

        Ok(Self {
            name: model_dir
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("qwen3-tts-model")
                .to_string(),
            package_root: model_dir.clone(),
            tokenizer_path,
            talker_config_path,
            talker_weights_path,
            generation_config: Qwen3TtsGenerationConfigSource::Path(generation_config_path),
            codec_config_path,
            codec_weights_path,
        })
    }
}

fn resolve_path(package_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        package_root.join(path)
    }
}

fn first_existing_file(root: &Path, relative_paths: &[&str]) -> Option<PathBuf> {
    relative_paths
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.is_file())
}

fn require_file(root: &Path, relative_path: &str) -> Result<PathBuf, Qwen3TtsLoadError> {
    let path = root.join(relative_path);
    if path.is_file() {
        Ok(path)
    } else {
        Err(Qwen3TtsLoadError::InvalidModelDir {
            message: format!(
                "{} is missing required file {}",
                root.display(),
                relative_path
            ),
        })
    }
}
