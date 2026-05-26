mod common;

use std::process::Command;

#[test]
fn python_prefill_oracle_is_available() {
    let model_dir = common::resolve_model_dir();
    let output = common::workspace_root().join("target/tmp/reference_v9_prefill.json");
    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "py/generate_reference_v9_prefill.py",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .current_dir(common::workspace_root())
        .status()
        .expect("failed to invoke Python prefill oracle");
    assert!(status.success(), "Python prefill oracle failed");
    assert!(output.is_file());
}
