mod common;

use burn::backend::LibTorch;

use tts_rs_qwen_burn::{
    VerificationArtifacts, load_qwen3_tts_talker, verify_qwen3_tts_talker_weights,
};

type TestBackend = LibTorch;

#[test]
#[ignore = "requires local Qwen weights and is slow"]
fn real_checkpoint_talker_weights_roundtrip() {
    let workspace_root = common::workspace_root();
    let model_dir = common::resolve_model_dir();
    let device = Default::default();

    let loaded =
        load_qwen3_tts_talker::<TestBackend>(&model_dir, &device).expect("checkpoint should load");
    assert_eq!(loaded.load_report.missing, 0);
    assert_eq!(loaded.load_report.unused, 0);
    assert_eq!(loaded.load_report.skipped, 0);

    let artifacts = VerificationArtifacts::new(
        workspace_root.join("artifacts/qwen3_tts/talker/test_roundtrip"),
    );
    let verification =
        verify_qwen3_tts_talker_weights(&loaded.model, &loaded.weights_path, Some(&artifacts))
            .expect("loaded model should roundtrip back to the original checkpoint");
    assert_eq!(verification.tensor_count, loaded.load_report.applied);
    assert_eq!(verification.tensor_count, 402);
}
