mod common;

use burn::backend::Flex;

use tts_rs_qwen_burn::{
    VerificationArtifacts, load_qwen3_tts_audio_codec, verify_qwen3_tts_audio_codec_weights,
};

type TestBackend = Flex;

#[test]
#[ignore = "requires local Qwen weights and is slow"]
fn real_checkpoint_audio_codec_weights_roundtrip() {
    let workspace_root = common::workspace_root();
    let model_dir = common::resolve_model_dir();
    let device = Default::default();

    let loaded = load_qwen3_tts_audio_codec::<TestBackend>(&model_dir, &device)
        .expect("audio codec checkpoint should load");
    assert_eq!(loaded.load_report.missing, 0);
    assert_eq!(loaded.load_report.skipped, 0);
    assert_eq!(loaded.load_report.applied, 496);

    let artifacts = VerificationArtifacts::new(
        workspace_root.join("artifacts/qwen3_tts/audio_codec/test_roundtrip"),
    );
    let verification =
        verify_qwen3_tts_audio_codec_weights(&loaded.model, &loaded.weights_path, Some(&artifacts))
            .expect("audio codec should roundtrip back to the source checkpoint");
    assert_eq!(verification.tensor_count, loaded.load_report.applied);
    assert_eq!(verification.tensor_count, 496);
}
