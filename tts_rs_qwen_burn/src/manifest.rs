use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_store::{
    BurnToPyTorchAdapter, KeyRemapper, ModuleSnapshot, ModuleStore, SafetensorsStore,
    TensorSnapshot,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Qwen3TtsVerifyError;
use crate::paths::ensure_parent_dir;

#[derive(Debug, Clone)]
pub struct LoadReport {
    pub applied: usize,
    pub skipped: usize,
    pub missing: usize,
    pub unused: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightManifestEntry {
    pub path: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightManifest {
    pub tensor_count: usize,
    pub entries: Vec<WeightManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightMismatch {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightComparisonReport {
    pub exact_match: bool,
    pub tensor_count: usize,
    pub source_tensor_count: usize,
    pub export_tensor_count: usize,
    pub missing_in_export: Vec<String>,
    pub missing_in_source: Vec<String>,
    pub mismatches: Vec<WeightMismatch>,
}

#[derive(Debug, Clone)]
pub struct VerificationArtifacts {
    pub source_manifest: PathBuf,
    pub export_manifest: PathBuf,
    pub comparison_report: PathBuf,
}

impl VerificationArtifacts {
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        let output_dir = output_dir.as_ref();
        Self {
            source_manifest: output_dir.join("source_manifest.json"),
            export_manifest: output_dir.join("rust_export_manifest.json"),
            comparison_report: output_dir.join("comparison_report.json"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightVerificationReport {
    pub tensor_count: usize,
    pub artifacts: Option<VerificationArtifacts>,
}

pub fn verify_module_weights<B: Backend, M: Module<B>>(
    model: &M,
    weights_path: impl AsRef<Path>,
    export_remapper: Option<KeyRemapper>,
    artifacts: Option<&VerificationArtifacts>,
) -> Result<WeightVerificationReport, Qwen3TtsVerifyError> {
    let source_manifest = build_source_manifest(weights_path.as_ref())?;
    let export_manifest = build_export_manifest(model, export_remapper)?;
    let comparison = compare_manifests(&source_manifest, &export_manifest);

    if let Some(artifacts) = artifacts {
        write_json(&source_manifest, &artifacts.source_manifest)?;
        write_json(&export_manifest, &artifacts.export_manifest)?;
        write_json(&comparison, &artifacts.comparison_report)?;
    }

    if !comparison.missing_in_export.is_empty() || !comparison.missing_in_source.is_empty() {
        return Err(Qwen3TtsVerifyError::KeySetMismatch {
            missing_in_export: comparison
                .missing_in_export
                .iter()
                .take(16)
                .cloned()
                .collect(),
            missing_in_source: comparison
                .missing_in_source
                .iter()
                .take(16)
                .cloned()
                .collect(),
        });
    }

    if let Some(mismatch) = comparison.mismatches.first() {
        return Err(Qwen3TtsVerifyError::TensorMismatch {
            path: mismatch.path.clone(),
            reason: mismatch.reason.clone(),
        });
    }

    Ok(WeightVerificationReport {
        tensor_count: comparison.tensor_count,
        artifacts: artifacts.cloned(),
    })
}

pub fn build_source_manifest(
    weights_path: impl AsRef<Path>,
) -> Result<WeightManifest, Qwen3TtsVerifyError> {
    let weights_path = weights_path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(&weights_path);
    let keys = store.keys().map_err(|source| Qwen3TtsVerifyError::Store {
        path: weights_path.clone(),
        source,
    })?;

    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let snapshot = store
            .get_snapshot(&key)
            .map_err(|source| Qwen3TtsVerifyError::Store {
                path: weights_path.clone(),
                source,
            })?
            .expect("key returned by keys() must exist");
        entries.push(snapshot_to_manifest_entry(&snapshot)?);
    }

    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(WeightManifest {
        tensor_count: entries.len(),
        entries,
    })
}

pub fn build_export_manifest<B: Backend, M: Module<B>>(
    model: &M,
    export_remapper: Option<KeyRemapper>,
) -> Result<WeightManifest, Qwen3TtsVerifyError> {
    let mut snapshots = model.collect(None, Some(Box::new(BurnToPyTorchAdapter)), true);
    if let Some(remapper) = export_remapper {
        let (remapped, _) = remapper.remap(std::mem::take(&mut snapshots));
        snapshots = remapped;
    }

    let mut entries = snapshots
        .iter()
        .map(snapshot_to_manifest_entry)
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(WeightManifest {
        tensor_count: entries.len(),
        entries,
    })
}

pub fn compare_manifests(
    source: &WeightManifest,
    export: &WeightManifest,
) -> WeightComparisonReport {
    let source_map: BTreeMap<&str, &WeightManifestEntry> = source
        .entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry))
        .collect();
    let export_map: BTreeMap<&str, &WeightManifestEntry> = export
        .entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry))
        .collect();

    let source_keys: BTreeSet<&str> = source_map.keys().copied().collect();
    let export_keys: BTreeSet<&str> = export_map.keys().copied().collect();

    let missing_in_export = source_keys
        .difference(&export_keys)
        .map(|path| (*path).to_string())
        .collect::<Vec<_>>();
    let missing_in_source = export_keys
        .difference(&source_keys)
        .map(|path| (*path).to_string())
        .collect::<Vec<_>>();

    let mut mismatches = Vec::new();
    for path in source_keys.intersection(&export_keys) {
        let source_entry = source_map
            .get(path)
            .expect("intersection keys must exist in source map");
        let export_entry = export_map
            .get(path)
            .expect("intersection keys must exist in export map");

        if source_entry.shape != export_entry.shape {
            mismatches.push(WeightMismatch {
                path: (*path).to_string(),
                reason: format!(
                    "shape mismatch: source={:?}, export={:?}",
                    source_entry.shape, export_entry.shape
                ),
            });
            continue;
        }

        if source_entry.dtype != export_entry.dtype {
            mismatches.push(WeightMismatch {
                path: (*path).to_string(),
                reason: format!(
                    "dtype mismatch: source={}, export={}",
                    source_entry.dtype, export_entry.dtype
                ),
            });
            continue;
        }

        if source_entry.sha256 != export_entry.sha256 {
            mismatches.push(WeightMismatch {
                path: (*path).to_string(),
                reason: format!(
                    "sha256 mismatch: source={}, export={}",
                    source_entry.sha256, export_entry.sha256
                ),
            });
        }
    }

    WeightComparisonReport {
        exact_match: missing_in_export.is_empty()
            && missing_in_source.is_empty()
            && mismatches.is_empty(),
        tensor_count: source.entries.len(),
        source_tensor_count: source.entries.len(),
        export_tensor_count: export.entries.len(),
        missing_in_export,
        missing_in_source,
        mismatches,
    }
}

pub fn write_json<T: Serialize>(
    value: &T,
    path: impl AsRef<Path>,
) -> Result<(), Qwen3TtsVerifyError> {
    let path = path.as_ref().to_path_buf();
    ensure_parent_dir(&path)?;
    let data = serde_json::to_vec_pretty(value).map_err(|source| Qwen3TtsVerifyError::Json {
        path: path.clone(),
        source,
    })?;
    fs::write(&path, data).map_err(|source| Qwen3TtsVerifyError::Io { path, source })
}

fn snapshot_to_manifest_entry(
    snapshot: &TensorSnapshot,
) -> Result<WeightManifestEntry, Qwen3TtsVerifyError> {
    let path = snapshot.full_path();
    let data = snapshot
        .to_data()
        .map_err(|err| Qwen3TtsVerifyError::TensorMismatch {
            path: path.clone(),
            reason: format!("failed to materialize tensor snapshot: {err}"),
        })?;

    Ok(WeightManifestEntry {
        path,
        shape: data.shape.as_slice().to_vec(),
        dtype: format!("{:?}", data.dtype),
        sha256: sha256_hex(data.as_bytes()),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

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

    fn entry(path: &str, shape: &[usize], dtype: &str, sha256: &str) -> WeightManifestEntry {
        WeightManifestEntry {
            path: path.to_string(),
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
            sha256: sha256.to_string(),
        }
    }

    fn manifest(entries: Vec<WeightManifestEntry>) -> WeightManifest {
        WeightManifest {
            tensor_count: entries.len(),
            entries,
        }
    }

    #[test]
    fn compare_manifests_reports_exact_match_for_identical_entries() {
        let source = manifest(vec![entry("a.weight", &[2, 3], "F32", "abc")]);
        let export = manifest(vec![entry("a.weight", &[2, 3], "F32", "abc")]);

        let report = compare_manifests(&source, &export);

        assert!(report.exact_match);
        assert_eq!(report.tensor_count, 1);
        assert!(report.missing_in_export.is_empty());
        assert!(report.missing_in_source.is_empty());
        assert!(report.mismatches.is_empty());
    }

    #[test]
    fn compare_manifests_reports_missing_keys_in_export() {
        let source = manifest(vec![
            entry("a.weight", &[2, 3], "F32", "abc"),
            entry("b.weight", &[3, 4], "F32", "def"),
        ]);
        let export = manifest(vec![entry("a.weight", &[2, 3], "F32", "abc")]);

        let report = compare_manifests(&source, &export);

        assert!(!report.exact_match);
        assert_eq!(report.missing_in_export, vec!["b.weight".to_string()]);
        assert!(report.missing_in_source.is_empty());
        assert!(report.mismatches.is_empty());
    }

    #[test]
    fn compare_manifests_reports_missing_keys_in_source() {
        let source = manifest(vec![entry("a.weight", &[2, 3], "F32", "abc")]);
        let export = manifest(vec![
            entry("a.weight", &[2, 3], "F32", "abc"),
            entry("extra.weight", &[3], "F32", "def"),
        ]);

        let report = compare_manifests(&source, &export);

        assert!(!report.exact_match);
        assert_eq!(report.missing_in_source, vec!["extra.weight".to_string()]);
        assert!(report.missing_in_export.is_empty());
        assert!(report.mismatches.is_empty());
    }

    #[test]
    fn compare_manifests_reports_shape_dtype_and_hash_mismatches() {
        let source = manifest(vec![
            entry("shape.weight", &[2, 3], "F32", "abc"),
            entry("dtype.weight", &[2, 3], "F32", "abc"),
            entry("hash.weight", &[2, 3], "F32", "abc"),
        ]);
        let export = manifest(vec![
            entry("shape.weight", &[3, 2], "F32", "abc"),
            entry("dtype.weight", &[2, 3], "F16", "abc"),
            entry("hash.weight", &[2, 3], "F32", "def"),
        ]);

        let report = compare_manifests(&source, &export);

        assert_eq!(report.mismatches.len(), 3);
        assert_eq!(report.mismatches[0].path, "dtype.weight");
        assert!(report.mismatches[0].reason.contains("dtype mismatch"));
        assert_eq!(report.mismatches[1].path, "hash.weight");
        assert!(report.mismatches[1].reason.contains("sha256 mismatch"));
        assert_eq!(report.mismatches[2].path, "shape.weight");
        assert!(report.mismatches[2].reason.contains("shape mismatch"));
    }

    #[test]
    fn verification_artifacts_use_expected_filenames() {
        let artifacts = VerificationArtifacts::new("/tmp/qwen-artifacts");

        assert_eq!(
            artifacts.source_manifest,
            PathBuf::from("/tmp/qwen-artifacts/source_manifest.json")
        );
        assert_eq!(
            artifacts.export_manifest,
            PathBuf::from("/tmp/qwen-artifacts/rust_export_manifest.json")
        );
        assert_eq!(
            artifacts.comparison_report,
            PathBuf::from("/tmp/qwen-artifacts/comparison_report.json")
        );
    }

    #[test]
    fn write_json_creates_parent_directories() {
        let temp_dir = TempDirGuard::new("manifest-write-json");
        let path = temp_dir.path.join("nested/report.json");
        let value = manifest(vec![entry("a.weight", &[1], "F32", "abc")]);

        write_json(&value, &path).expect("json should be written");

        assert!(path.is_file());
        let parsed: Value =
            serde_json::from_slice(&fs::read(&path).expect("json should exist")).unwrap();
        assert_eq!(parsed["tensor_count"], 1);
        assert_eq!(parsed["entries"][0]["sha256"], "abc");
    }
}
