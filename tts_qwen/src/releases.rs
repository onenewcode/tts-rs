use tts_core::TtsCoreError;

use crate::arch::{QWEN3_TTS_ARCH, QwenArchitectureDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenProfile {
    Base,
    CustomVoice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenReleaseId {
    Qwen3Tts12Hz06BBase,
    Qwen3Tts12Hz06BCustomVoice,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenReleaseManifest {
    pub(crate) id: QwenReleaseId,
    pub(crate) label: &'static str,
    pub(crate) architecture: &'static QwenArchitectureDescriptor,
    pub(crate) profile: QwenProfile,
}

const RELEASES: &[QwenReleaseManifest] = &[
    QwenReleaseManifest {
        id: QwenReleaseId::Qwen3Tts12Hz06BBase,
        label: "qwen3-tts-12hz-0.6b-base",
        architecture: &QWEN3_TTS_ARCH,
        profile: QwenProfile::Base,
    },
    QwenReleaseManifest {
        id: QwenReleaseId::Qwen3Tts12Hz06BCustomVoice,
        label: "qwen3-tts-12hz-0.6b-customvoice",
        architecture: &QWEN3_TTS_ARCH,
        profile: QwenProfile::CustomVoice,
    },
];

pub(crate) fn parse_release_manifest(
    value: &str,
) -> Result<&'static QwenReleaseManifest, TtsCoreError> {
    RELEASES
        .iter()
        .find(|release| release.label == value)
        .ok_or_else(|| {
            let supported = RELEASES
                .iter()
                .map(|release| release.label)
                .collect::<Vec<_>>()
                .join(", ");
            TtsCoreError::Config {
                message: format!(
                    "unsupported qwen variant `{value}`; currently supported: {supported}"
                ),
            }
        })
}

#[cfg(test)]
mod tests {
    use super::{QwenProfile, QwenReleaseId, parse_release_manifest};
    use crate::arch::QwenArchitectureId;

    #[test]
    fn parses_supported_releases() {
        let base = parse_release_manifest("qwen3-tts-12hz-0.6b-base").unwrap();
        assert_eq!(base.id, QwenReleaseId::Qwen3Tts12Hz06BBase);
        assert_eq!(base.profile, QwenProfile::Base);
        assert_eq!(base.architecture.id, QwenArchitectureId::Qwen3Tts);
        assert_eq!(base.architecture.label, "qwen3_tts");

        let custom = parse_release_manifest("qwen3-tts-12hz-0.6b-customvoice").unwrap();
        assert_eq!(custom.id, QwenReleaseId::Qwen3Tts12Hz06BCustomVoice);
        assert_eq!(custom.profile, QwenProfile::CustomVoice);
        assert_eq!(custom.architecture.id, QwenArchitectureId::Qwen3Tts);
    }
}
