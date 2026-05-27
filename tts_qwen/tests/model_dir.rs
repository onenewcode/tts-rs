mod common;

#[test]
#[should_panic(expected = "QWEN_TTS_MODEL_DIR must be set")]
fn resolve_model_dir_requires_explicit_env_var() {
    let _ = common::resolve_model_dir();
}
