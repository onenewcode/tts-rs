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

pub(crate) fn release_for_profile(profile: QwenProfile) -> &'static QwenReleaseManifest {
    match profile {
        QwenProfile::Base => &RELEASES[0],
        QwenProfile::CustomVoice => &RELEASES[1],
    }
}

#[cfg(test)]
mod tests {
    use super::{QwenProfile, QwenReleaseId, release_for_profile};
    use crate::arch::QwenArchitectureId;

    #[test]
    fn resolves_releases_for_profiles() {
        let base = release_for_profile(QwenProfile::Base);
        assert_eq!(base.id, QwenReleaseId::Qwen3Tts12Hz06BBase);
        assert_eq!(base.profile, QwenProfile::Base);
        assert_eq!(base.architecture.id, QwenArchitectureId::Qwen3Tts);
        assert_eq!(base.architecture.label, "qwen3_tts");

        let custom = release_for_profile(QwenProfile::CustomVoice);
        assert_eq!(custom.id, QwenReleaseId::Qwen3Tts12Hz06BCustomVoice);
        assert_eq!(custom.profile, QwenProfile::CustomVoice);
        assert_eq!(custom.architecture.id, QwenArchitectureId::Qwen3Tts);
    }
}
