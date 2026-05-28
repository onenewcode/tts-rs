use std::path::{Path, PathBuf};

use crate::Qwen3TtsLoadError;

use super::manifest::{
    Qwen3TtsPackageManifest, Qwen3TtsProfileManifest, Qwen3TtsProfilesManifest,
};

const PACKAGE_MANIFEST_FORMAT: &str = "qwen3_tts_package/v1";
const DEFAULT_PACKAGE_MANIFEST_NAME: &str = "qwen3_tts_package.yaml";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen3TtsPackageSource {
    ManifestPath(PathBuf),
    PackageDir(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsPackage {
    pub package_root: PathBuf,
    pub name: String,
    pub tokenizer_path: PathBuf,
    pub talker_config_path: PathBuf,
    pub talker_weights_path: PathBuf,
    pub codec_config_path: PathBuf,
    pub codec_weights_path: PathBuf,
    pub profiles: Qwen3TtsPackageProfiles,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsPackageProfiles {
    pub base: Option<Qwen3TtsProfilePackage>,
    pub custom_voice: Option<Qwen3TtsProfilePackage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsProfilePackage {
    pub generation_config_path: PathBuf,
    pub control_config_path: PathBuf,
}

impl Qwen3TtsPackage {
    pub fn load(source: &Qwen3TtsPackageSource) -> Result<Self, Qwen3TtsLoadError> {
        let manifest_path = source.manifest_path();
        let package_root = manifest_path.parent().ok_or_else(|| Qwen3TtsLoadError::InvalidManifest {
            message: format!("package manifest path has no parent: {}", manifest_path.display()),
        })?.to_path_buf();

        let raw = std::fs::read_to_string(&manifest_path).map_err(|source| Qwen3TtsLoadError::Io {
            path: manifest_path.clone(),
            source,
        })?;
        let manifest: Qwen3TtsPackageManifest = serde_yaml::from_str(&raw).map_err(|source| {
            Qwen3TtsLoadError::ManifestParse {
                path: manifest_path.clone(),
                source,
            }
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
            codec_config_path: resolve_path(&package_root, &manifest.artifacts.codec_config),
            codec_weights_path: resolve_path(&package_root, &manifest.artifacts.codec_weights),
            profiles: normalize_profiles(&package_root, manifest.profiles),
        })
    }
}

impl Qwen3TtsPackageSource {
    fn manifest_path(&self) -> PathBuf {
        match self {
            Self::ManifestPath(path) => path.clone(),
            Self::PackageDir(path) => path.join(DEFAULT_PACKAGE_MANIFEST_NAME),
        }
    }
}

fn normalize_profiles(
    package_root: &Path,
    profiles: Qwen3TtsProfilesManifest,
) -> Qwen3TtsPackageProfiles {
    Qwen3TtsPackageProfiles {
        base: normalize_profile(package_root, profiles.base),
        custom_voice: normalize_profile(package_root, profiles.custom_voice),
    }
}

fn normalize_profile(
    package_root: &Path,
    profile: Option<Qwen3TtsProfileManifest>,
) -> Option<Qwen3TtsProfilePackage> {
    profile.map(|profile| Qwen3TtsProfilePackage {
        generation_config_path: resolve_path(package_root, &profile.generation_config),
        control_config_path: resolve_path(package_root, &profile.control_config),
    })
}

fn resolve_path(package_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        package_root.join(path)
    }
}
