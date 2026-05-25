use std::path::PathBuf;

use burn::backend::Flex;
use tts_rs_qwen_burn::{
    VerificationArtifacts, default_workspace_root, find_local_qwen_tts_model_dir,
    load_qwen3_tts_speech_tokenizer, verify_qwen3_tts_speech_tokenizer_weights,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        let workspace_root = default_workspace_root();
        find_local_qwen_tts_model_dir(workspace_root).expect("local qwen model directory")
    });
    let output_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_workspace_root().join("artifacts/qwen3_tts/speech_tokenizer"));

    let device = Default::default();
    let loaded = load_qwen3_tts_speech_tokenizer::<Flex>(&model_dir, &device)?;
    let artifacts = VerificationArtifacts::new(output_dir);
    let verification = verify_qwen3_tts_speech_tokenizer_weights(
        &loaded.model,
        &loaded.weights_path,
        Some(&artifacts),
    )?;

    println!("model_dir: {}", loaded.model_dir.display());
    println!("weights: {}", loaded.weights_path.display());
    println!(
        "load_report: applied={}, skipped={}, missing={}, unused={}",
        loaded.load_report.applied,
        loaded.load_report.skipped,
        loaded.load_report.missing,
        loaded.load_report.unused
    );
    println!(
        "verification: {} tensors matched exactly",
        verification.tensor_count
    );
    println!(
        "artifacts: source_manifest={}, export_manifest={}, comparison_report={}",
        artifacts.source_manifest.display(),
        artifacts.export_manifest.display(),
        artifacts.comparison_report.display()
    );

    Ok(())
}
